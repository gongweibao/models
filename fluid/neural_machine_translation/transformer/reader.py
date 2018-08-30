import glob
import os
import random
import tarfile
import cPickle

import multiprocessing
import math
import copy

class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class Converter(object):
    def __init__(self, vocab, beg, end, unk, delimiter):
        self._vocab = vocab
        self._beg = beg
        self._end = end
        self._unk = unk
        self._delimiter = delimiter

    def __call__(self, sentence):
        return [self._beg] + [
            self._vocab.get(w, self._unk)
            for w in sentence.split(self._delimiter)
        ] + [self._end]


class ComposedConverter(object):
    def __init__(self, converters):
        self._converters = converters

    def __call__(self, parallel_sentence):
        return [
            self._converters[i](parallel_sentence[i])
            for i in range(len(self._converters))
        ]


class SentenceBatchCreator(object):
    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp


class TokenBatchCreator(object):
    def __init__(self, batch_size):
        self.batch = []
        self.max_len = -1
        self._batch_size = batch_size

    def append(self, info):
        cur_len = info.max_len
        max_len = max(self.max_len, cur_len)
        if max_len * (len(self.batch) + 1) > self._batch_size:
            result = self.batch
            self.batch = [info]
            self.max_len = cur_len
            return result
        else:
            self.max_len = max_len
            self.batch.append(info)


class SampleInfo(object):
    def __init__(self, i, max_len, min_len):
        self.i = i
        self.min_len = min_len
        self.max_len = max_len


class MinMaxFilter(object):
    def __init__(self, max_len, min_len, underlying_creator):
        self._min_len = min_len
        self._max_len = max_len
        self._creator = underlying_creator

    def append(self, info):
        if info.max_len > self._max_len or info.min_len < self._min_len:
            return
        else:
            return self._creator.append(info)

    @property
    def batch(self):
        return self._creator.batch


class DataReader(object):
    """
    The data reader loads all data from files and produces batches of data
    in the way corresponding to settings.

    An example of returning a generator producing data batches whose data
    is shuffled in each pass and sorted in each pool:

    ```
    config = ReaderConfig()
    config.src_vocab_fpath='data/src_vocab_file',
    config.trg_vocab_fpath='data/trg_vocab_file',
    config.fpattern='data/part-*',
    config.use_token_batch=True,
    config.batch_size=2000,
    config.pool_size=10000,
    config.sort_type=SortType.POOL,

    reader = DataReader(config=config)
    reader.load_data()
    train_data = reader.batch_generator()
    ```
    """


    def __init__(self, config):
        self._config = config
        self._random = random.Random(x=config.seed)
        self._sample_infos = []

        self._src_vocab = self.load_dict(self._config.src_vocab_fpath)
        self._only_src = True
        if self._config.trg_vocab_fpath is not None:
            self._trg_vocab = self.load_dict(self._config.trg_vocab_fpath)
            self._only_src = False

        self._src_seq_ids = []
        self._trg_seq_ids = None if self._only_src else []
        self._sample_infos = []

    def load_data(self):
        if isinstance(self._config.fpattern, list):
            fpaths = self._config.fpattern
        else:
            print("fpattren:", self._config.fpattern)
            fpaths = glob.glob(self._config.fpattern)
        assert len(fpaths) > 0, "no input files"

        q = multiprocessing.Queue()
        if self._config.process_num <= 0:
            processes = multiprocessing.cpu_count()
        else:
            processes = self.process_num
        size = int(math.ceil(float(len(fpaths)) / processes))

        procs=[]
        for i in range(processes):
            conf = copy.deepcopy(self._config)
            conf.fpattern = fpaths[i * size:(i + 1) * size]
            p = multiprocessing.Process(target=load_data_in_process, args=(conf, q))
            procs.append(p)

        for i in range(processes):
            procs[i].start()
        
        done_num = 0
        idx=0
        while True:
            if done_num >= processes:
                print("recv one done")
                break
            src_trg_ids = q.get()

            if src_trg_ids is None:
                done_num += 1
                continue

            self._src_seq_ids.append(src_trg_ids[0])
            lens = [len(src_trg_ids[0])]
            if not self._only_src:
                self._trg_seq_ids.append(src_trg_ids[1])
                lens.append(len(src_trg_ids[1]))
            self._sample_infos.append(SampleInfo(idx, max(lens), min(lens)))
            idx+=1

        for i in range(processes):
            procs[i].join()

    def _load_src_trg_ids(self, q):
        converters = [
            Converter(
                vocab=self._src_vocab,
                beg=self._src_vocab[self._config.start_mark],
                end=self._src_vocab[self._config.end_mark],
                unk=self._src_vocab[self._config.unk_mark],
                delimiter=self._config.token_delimiter)
        ]
        if not self._only_src:
            converters.append(
                Converter(
                    vocab=self._trg_vocab,
                    beg=self._trg_vocab[self._config.start_mark],
                    end=self._trg_vocab[self._config.end_mark],
                    unk=self._trg_vocab[self._config.unk_mark],
                    delimiter=self._config.token_delimiter))

        converters = ComposedConverter(converters)

        for i, line in enumerate(self._load_lines(self._config.fpattern, self._config.tar_fname)):
            if len(line) <= 0:
                continue

            src_trg_ids = converters(line)
            q.put(src_trg_ids)
        q.put(None)

    def get_sample_infos(self):
        return self._sample_infos

    def _load_lines(self, fpattern, tar_fname):
        if isinstance(fpattern, list):
            fpaths = fpattern
        else:
            fpaths = glob.glob(fpattern)

        if len(fpaths) == 1 and tarfile.is_tarfile(fpaths[0]):
            if tar_fname is None:
                raise Exception("If tar file provided, please set tar_fname.")

            f = tarfile.open(fpaths[0], "r")
            for line in f.extractfile(tar_fname):
                fields = line.strip("\n").split(self._config.field_delimiter)
                if (not self._only_src and len(fields) == 2) or (
                        self._only_src and len(fields) == 1):
                    yield fields
        else:
            for fpath in fpaths:
                print("open file:", fpath)
                if not os.path.isfile(fpath):
                    raise IOError("Invalid file: %s" % fpath)

                with open(fpath, "r") as f:
                    for line in f:
                        fields = line.strip("\n").split(self._config.field_delimiter)
                        if (not self._only_src and len(fields) == 2) or (
                                self._only_src and len(fields) == 1):
                            yield fields

    @staticmethod
    def load_dict(dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "r") as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip("\n")
                else:
                    word_dict[line.strip("\n")] = idx
        return word_dict

    def batch_generator(self):
        # global sort or global shuffle
        if self._config.sort_type == SortType.GLOBAL:
            infos = sorted(
                self._sample_infos, key=lambda x: x.max_len, reverse=True)
        else:
            if self._config._shuffle:
                infos = self._sample_infos
                self._random.shuffle(infos)
            else:
                infos = self._sample_infos

            if self._config.sort_type == SortType.POOL:
                for i in range(0, len(infos), self._cofig.pool_size):
                    infos[i:i + self._config.pool_size] = sorted(
                        infos[i:i + self._config.pool_size], key=lambda x: x.max_len)

        # concat batch
        batches = []
        batch_creator = TokenBatchCreator(
            self._config.batch_size
        ) if self._config.use_token_batch else SentenceBatchCreator(self._config.batch_size)
        batch_creator = MinMaxFilter(self._config.max_length, self._config.min_length,
                                     batch_creator)

        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batches.append(batch)

        if not self._config.clip_last_batch and len(batch_creator.batch) != 0:
            batches.append(batch_creator.batch)

        if self._config.shuffle_batch:
            self._random.shuffle(batches)

        for batch in batches:
            batch_ids = [info.i for info in batch]

            if self._only_src:
                yield [[self._src_seq_ids[idx]] for idx in batch_ids]
            else:
                yield [(self._src_seq_ids[idx], self._trg_seq_ids[idx][:-1],
                        self._trg_seq_ids[idx][1:]) for idx in batch_ids]

class ReaderConfig(object):
    """
    Attributes:
        src_vocab_fpath(str): The path of vocabulary file of source language.
        trg_vocab_fpath(str): The path of vocabulary file of target language.
        fpattern(str|list): The pattern to match data files.
        batch_size(int): The number of sequences contained in a mini-batch or the maximum number of tokens (include paddings) contained in a mini-batch.
        pool_size(int): The size of pool buffer.
        sort_type(str): The grain to sort by length: 'global' for all instances; 'pool' for instances in pool; 'none' for no sort.
        clip_last_batch(bool): Whether to clip the last uncompleted batch.
        tar_fname(str): The data file in tar if fpattern matches a tar file.
        min_length(int): The minimum length used to filt sequences.
        max_length(int): The maximum length used to filt sequences.
        shuffle(bool): Whether to shuffle all instances.
        shuffle_batch(bool): Whether to shuffle the generated batches.
        use_token_batch(bool): Whether to produce batch data according to token number.
        field_delimiter(str): The delimiter used to split source and target in each line of data file.
        token_delimiter(str): The delimiter used to split tokens in source or target sentences.
        start_mark(str): The token representing for the beginning of sentences in dictionary.
        end_mark(str): The token representing for the end of sentences in dictionary.
        unk_mark(str): The token representing for unknown word in dictionary.
        seed(int): The seed for random.
    """

    src_vocab_fpath = None
    trg_vocab_fpath = None
    fpattern        = None
    batch_size      = None
    pool_size       = None
    sort_type       = SortType.GLOBAL,
    clip_last_batch = True
    tar_fname       = None
    min_length      = 0
    max_length      = 100
    shuffle         = True
    shuffle_batch   = True
    use_token_batch = True
    field_delimiter = "\t"
    token_delimiter = " "
    start_mark      = "<s>"
    end_mark        = "<e>"
    unk_mark        = "<unk>"
    seed            = 0
    process_num     = 0


"""
class MultiProcessReader(object):
    def __init__(self, config):
        if isinstance(config.fpattern, list):
            fpaths = config.fpattern
        else:
            fpaths = glob.glob(config.fpattern)

        assert len(fpaths) > 0, "no input files"
        #print(sorted(fpaths))

        processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processes)
        size = int(math.ceil(float(len(fpaths)) / processes))

        configs = []
        for i in range(processes):
            conf = copy.deepcopy(config)
            conf.fpattern = fpaths[i * size:(i + 1) * size]
            configs.append(conf)

        rets = pool.map(load_data_in_process, configs)
        for i in range(processes):
            if rets[i] is None:
                continue
            print(i, len(rets[i].get_sample_infos()))
"""

def load_data_in_process(config, q):
    if len(config.fpattern) < 1:
        print("task set without input files so return")
        q.put(None)
        return

    reader=DataReader(config)
    reader._load_src_trg_ids(q)
    print("proc {} complete".format(config.fpattern))
    return


