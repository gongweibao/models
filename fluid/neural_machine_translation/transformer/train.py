import os
import time
import argparse
import ast
import numpy as np
import multiprocessing
import sys

import paddle
import paddle.fluid as fluid

from model import transformer, position_encoding_init
from optim import LearningRateScheduler
from config import *
import reader
from paddle.fluid import debugger
import paddle.fluid.layers as layers

import hashlib

def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--train_file_pattern",
        type=str,
        required=True,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        help="The pattern to match validation data files.")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--pass_num",
        type=int,
        default=TrainTaskConfig.pass_num,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=10000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--local',
        type=ast.literal_eval,
        default=True,
        help='Whether to run as local mode.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")

    parser.add_argument(
        '--check_acc', type=ast.literal_eval, default=False, help="check only accuracy.")

    args = parser.parse_args()
    # Append args related to dict
    src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
        str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
        "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
        str(src_dict[args.special_token[2]])
    ]
    merge_cfg_from_list(args.opts + dict_args,
                        [TrainTaskConfig, ModelHyperParams])
    return args


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    num_token = reduce(lambda x, y: x + y,
                       [len(inst) for inst in insts]) if return_num_token else 0
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            range(1, len(inst) + 1) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, data_input_names, util_input_names, src_pad_idx,
                        trg_pad_idx, n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    # These shape tensors are used in reshape_op.
    src_data_shape = np.array([-1, src_max_len, d_model], dtype="int32")
    trg_data_shape = np.array([-1, trg_max_len, d_model], dtype="int32")
    src_slf_attn_pre_softmax_shape = np.array(
        [-1, src_slf_attn_bias.shape[-1]], dtype="int32")
    src_slf_attn_post_softmax_shape = np.array(
        [-1] + list(src_slf_attn_bias.shape[1:]), dtype="int32")
    trg_slf_attn_pre_softmax_shape = np.array(
        [-1, trg_slf_attn_bias.shape[-1]], dtype="int32")
    trg_slf_attn_post_softmax_shape = np.array(
        [-1] + list(trg_slf_attn_bias.shape[1:]), dtype="int32")
    trg_src_attn_pre_softmax_shape = np.array(
        [-1, trg_src_attn_bias.shape[-1]], dtype="int32")
    trg_src_attn_post_softmax_shape = np.array(
        [-1] + list(trg_src_attn_bias.shape[1:]), dtype="int32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
            trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
        ]))
    util_input_dict = dict(
        zip(util_input_names, [
            src_data_shape, src_slf_attn_pre_softmax_shape,
            src_slf_attn_post_softmax_shape, trg_data_shape,
            trg_slf_attn_pre_softmax_shape, trg_slf_attn_post_softmax_shape,
            trg_src_attn_pre_softmax_shape, trg_src_attn_post_softmax_shape
        ]))
    return data_input_dict, util_input_dict, np.asarray(
        [num_token], dtype="float32")


def read_multiple(reader, count, clip_last=True):
    """
    Stack data from reader for multi-devices.
    """

    def __impl__():
        res = []
        for item in reader():
            res.append(item)
            if len(res) == count:
                yield res
                res = []
        if len(res) == count:
            yield res
        elif not clip_last:
            data = []
            for item in res:
                data += item
            if len(data) > count:
                inst_num_per_part = len(data) // count
                yield [
                    data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                    for i in range(count)
                ]

    return __impl__


def split_data(data, num_part):
    """
    Split data for each device.
    """
    if len(data) == num_part:
        return data
    data = data[0]
    #print "data:", data
    inst_num_per_part = len(data) // num_part
    #print "data len:", len(data), " inst_num_per_part:", inst_num_per_part
    return [
        data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
        for i in range(num_part)
    ]


def test_context(train_progm, avg_cost, train_exe, dev_count, data_input_names,
                 util_input_names, sum_cost, token_num):
    # Context to do validation.
    test_program = train_progm.clone()
    with fluid.program_guard(test_program):
        test_program = fluid.io.get_inference_program([avg_cost])

    val_data = reader.DataReader(
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        fpattern=args.val_file_pattern,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else dev_count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False,
        shuffle=False,
        shuffle_batch=False)

    test_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=test_program,
        share_vars_from=train_exe)

    def test(exe=test_exe):
        test_total_cost = 0
        test_total_token = 0
        test_data = read_multiple(
            reader=val_data.batch_generator,
            count=dev_count if args.use_token_batch else 1)
        for batch_id, data in enumerate(test_data()):
            feed_list = []
            for place_id, data_buffer in enumerate(
                    split_data(
                        data, num_part=dev_count)):
                data_input_dict, util_input_dict, _ = prepare_batch_input(
                    data_buffer, data_input_names, util_input_names,
                    ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head, ModelHyperParams.d_model)
                feed_list.append(
                    dict(data_input_dict.items() + util_input_dict.items()))

            outs = exe.run(feed=feed_list,
                           fetch_list=[sum_cost.name, token_num.name])
            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            test_total_cost += sum_cost_val.sum()
            test_total_token += token_num_val.sum()
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    return test

def _clone_var_in_block_(block, var):
    assert isinstance(var, fluid.Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)

def train_loop(exe, train_progm, dev_count, sum_cost, avg_cost, lr_scheduler,
               token_num, predict):
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        fluid.io.load_persistables(exe, TrainTaskConfig.ckpt_path)
        lr_scheduler.current_steps = TrainTaskConfig.start_step
    else:
        """
        dis_program = fluid.Program()
        dis_block = dis_program.global_block()
        for each_var in fluid.framework.default_startup_program().list_vars():
            # NOTE: don't save the variable which type is RAW
            if each_var.type == fluid.core.VarDesc.VarType.RAW:
                continue
            print each_var.name
            new_var = _clone_var_in_block_(dis_block, each_var)
            layers.Print(input=new_var, summarize=10, program=dis_program)

        #print "init dis_program"
        #debugger.draw_block_graphviz(dis_program.global_block(), path="dis.dot")

        exe.run(dis_program)
        exe.run(fluid.framework.default_startup_program())
        """

        fetch_list=[]
        for var in fluid.framework.default_startup_program().list_vars():
            if "fc_75.w_0" in var.name or "moment1_93" in var.name or "moment2_93" in var.name:
                fetch_list.append(var.name)
        outs = exe.run(fluid.framework.default_startup_program(), fetch_list=fetch_list)
        assert(len(fetch_list) == len(outs))
        for name, value in list(zip(fetch_list, outs)):
            print name, value

        #exe.run(fluid.framework.default_startup_program())

    train_data = reader.DataReader(
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        fpattern=args.train_file_pattern,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else dev_count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False)
    train_data = read_multiple(
        reader=train_data.batch_generator,
        count=dev_count if args.use_token_batch else 1)

    build_strategy = fluid.BuildStrategy()
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=sum_cost.name,
        main_program=train_progm,
        build_strategy=build_strategy)

    data_input_names = encoder_data_input_fields + decoder_data_input_fields[:
                                                                             -1] + label_data_input_fields
    util_input_names = encoder_util_input_fields + decoder_util_input_fields

    if args.val_file_pattern is not None:
        test = test_context(train_progm, avg_cost, train_exe, dev_count,
                            data_input_names, util_input_names, sum_cost,
                            token_num)

    init = False
    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        #print "train_data len:", len(train_data())
        for batch_id, data in enumerate(train_data()):
            feed_list = []
            total_num_token = 0
            if args.local:
                lr_rate = lr_scheduler.update_learning_rate()
                print "pass_id:", pass_id, "batch_id:", batch_id, "step:", lr_scheduler.current_steps, ", learning_rate:", lr_rate
            for place_id, data_buffer in enumerate(
                    split_data(
                        data, num_part=dev_count)):
                if args.local:
                    a0=[]
                    a1=[]
                    for i,x in enumerate(data_buffer):
                        if i % 2 == 0:
                            a0.append(x)
                            continue
                        a1.append(x)

                    value=''
                    for t in a0:
                        for n in t:
                            tmp = [str(i) for i in n]
                            #print tmp
                            value += ''.join(tmp)
                    m = hashlib.md5()
                    m.update(value)
                    a0_m = m.hexdigest()

                    value=''
                    for t in a1:
                        for n in t:
                            tmp = [str(i) for i in n]
                            value += ''.join(tmp)
                    m = hashlib.md5()
                    m.update(value)
                    a1_m = m.hexdigest()

                    #print "a0:", a0
                    #print "a1:", a1
                    print "batch_id:", batch_id, ", a0:", a0_m, ", a1:", a1_m
                else:
                    a=[]
                    for i,x in enumerate(data_buffer):
                        a.append(x)

                    value=''
                    for t in a:
                        for n in t:
                            tmp = [str(i) for i in n]
                            value += ''.join(tmp)
                    m = hashlib.md5()
                    m.update(value)
                    a_m = m.hexdigest()

                    #print "a:", a
                    print "batch_id:", batch_id, ", a:", a_m

                data_input_dict, util_input_dict, num_token = prepare_batch_input(
                    data_buffer, data_input_names, util_input_names,
                    ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head, ModelHyperParams.d_model)
                total_num_token += num_token
                feed_kv_pairs = data_input_dict.items() + util_input_dict.items(
                )
                if args.local:
                    feed_kv_pairs += {
                        lr_scheduler.learning_rate.name: lr_rate
                    }.items()
                feed_list.append(dict(feed_kv_pairs))

                if not init:
                    for pos_enc_param_name in pos_enc_param_names:
                        pos_enc = position_encoding_init(
                            ModelHyperParams.max_length + 1,
                            ModelHyperParams.d_model)
                        feed_list[place_id][pos_enc_param_name] = pos_enc
                #print batch_id, place_id, feed_list
            if not args.check_acc:
                for feed_dict in feed_list:
                    feed_dict[sum_cost.name + "@GRAD"] = 1. / total_num_token
            else:
                print "batch_size:", args.batch_size
                b = 100*args.batch_size
                a  = np.asarray([b], dtype="float32")
                for feed_dict in feed_list:
                    feed_dict[sum_cost.name + "@GRAD"] = 1. / a
                print "total_num_token:",a 

            if args.local:
                outs = train_exe.run(fetch_list=[sum_cost.name, token_num.name, "fc_75.w_0", "fc_75.w_0@GRAD", "moment1_93", "moment2_93", "learning_rate"],
                                     feed=feed_list)
                print "batch_id:", batch_id, ", fc_75.w_0", outs[2]
                print "fc_75.w_0@GRAD", outs[3]
                print "moment1_93", outs[4]
                print "moment2_93", outs[5]
                print "learning_rate", outs[6]

            else:
                outs = train_exe.run(fetch_list=[sum_cost.name, token_num.name, "fc_75.w_0"],
                                     feed=feed_list)
                print "batch_id:", batch_id, ", fc_75.w_0", outs[2]

            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            total_sum_cost = sum_cost_val.sum(
            )  # sum the cost from multi-devices
            total_token_num = token_num_val.sum()
            total_avg_cost = total_sum_cost / total_token_num
            print("epoch: %d, batch: %d, sum loss: %f, avg loss: %f, ppl: %f" %
                  (pass_id, batch_id, total_sum_cost, total_avg_cost,
                   np.exp([min(total_avg_cost, 100)])))
            init = True
            #if batch_id == 2:
            #    break
        #break
        # Validate and save the model for inference.
        print("epoch: %d, " % pass_id +
              ("val avg loss: %f, val ppl: %f, " % test()
               if args.val_file_pattern is not None else "") + "consumed %fs" %
              (time.time() - pass_start_time))
        fluid.io.save_persistables(
            exe,
            os.path.join(TrainTaskConfig.ckpt_dir,
                         "pass_" + str(pass_id) + ".checkpoint"))
        fluid.io.save_inference_model(
            os.path.join(TrainTaskConfig.model_dir,
                         "pass_" + str(pass_id) + ".infer.model"),
            data_input_names[:-2] + util_input_names, [predict], exe)


def train(args):
    # priority: ENV > args > config
    is_local = os.getenv("PADDLE_IS_LOCAL", "1")
    if is_local == '0':
        args.local = False
    print args

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")

    if training_role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    exe = fluid.Executor(place)


    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.dropout,
        ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps)
    lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                         TrainTaskConfig.warmup_steps,
                                         TrainTaskConfig.learning_rate)

    if args.local:
        #print "print forward network"
        #debugger.draw_block_graphviz(fluid.framework.default_main_program().global_block(), path="local_main.dot")
        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_scheduler.learning_rate,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        print "forward startup:", fluid.framework.default_startup_program()
        optimizer.minimize(sum_cost)
        print "forward main:", fluid.framework.default_main_program()
        #fluid.memory_optimize(fluid.default_main_program())
    elif args.sync == False:
        optimizer = fluid.optimizer.SGD(0.003)
        optimizer.minimize(sum_cost)
    else:
        lr_decay = fluid.layers\
         .learning_rate_scheduler\
         .noam_decay(ModelHyperParams.d_model,
            TrainTaskConfig.warmup_steps)

        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_decay,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        optimizer.minimize(sum_cost)

    if args.local:
        print("local start_up:")
        #print "local main:", fluid.default_main_program()
        train_loop(exe,
                   fluid.default_main_program(), dev_count, sum_cost, avg_cost,
                   lr_scheduler, token_num, predict)
    else:
        ps_ports = os.getenv("PADDLE_PORT", "6174")     # port or port,port...
        ps_ips = os.getenv("PADDLE_PSERVERS")           #         ip,ip...

        ps_ips = ps_ips.split(",")
        ps_ports = ps_ports.split(",")

        eplist = []
        current_endpoint=""
        if len(ps_ports) > 1:
            assert(len(ps_ips) == len(ps_ports))
            for ip,port in zip(ps_ips, ps_ports):
                eplist.append(':'.join([ip, port]))
        else:
            for ip in ps_ips:
                eplist.append(':'.join([ip, ps_ports[0]]))
        print("eplist:", eplist)
        pserver_endpoints = ",".join(eplist)         # ip:port,ip:port...
        print("pserver_endpoints:", pserver_endpoints)

        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        #if args.check_acc:
            #assert(trainers == 2)
            #assert(len(ps_ips) == 2)
        
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = False
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)

        if training_role == "PSERVER":
            current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
            print("current_endpoints:", current_endpoint)
            if not current_endpoint:
                print("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)

            print "psserver begin run"
            fetch_list=[]
            for var in pserver_startup.list_vars():
                if "fc_75.w_0" in var.name and "@" not in var.name:
                    fetch_list.append(var.name)
            outs = exe.run(pserver_startup, fetch_list=fetch_list)
            assert(len(fetch_list) == len(outs))
            for name, value in list(zip(fetch_list, outs)):
                print name, value

            #print "pserver startup:", pserver_startup
            print "blocks num:", len(pserver_prog.blocks)
            for b in range(1, len(pserver_prog.blocks)):
                print "block:", b, pserver_prog.blocks[b] 
            #print pserver_prog
            #for b in pserver_prog.blocks:
            #    debugger.draw_block_graphviz(dis_program.global_block(), path="dis.dot")

            #exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            print "trainer begin run"
            trainer_prog = t.get_trainer_program()
            train_loop(exe, trainer_prog, dev_count, sum_cost, avg_cost,
                       lr_scheduler, token_num, predict)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")


if __name__ == "__main__":
    args = parse_args()
    if args.check_acc:
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        ModelHyperParams.dropout = 0.0

        assert(not args.shuffle)
        assert(not args.shuffle_batch)
        assert(not args.use_token_batch)

    train(args)
