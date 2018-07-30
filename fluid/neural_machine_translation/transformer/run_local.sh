export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/:/workspace/brpc
export PYTHONPATH=$PYTHONPATH:/paddle/build/build_develop/python
#export CPU_NUM=1

FLAGS_benchmark=true CUDA_VISIBLE_DEVICES=0 FLAGS_fraction_of_gpu_memory_to_use=0.2 GLOG_v=1 GLOG_logtostderr=1 python -u train.py \
  --src_vocab_fpath test_data/vocab.bpe.32000 \
  --trg_vocab_fpath test_data/vocab.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern test_data/train.tok.clean.bpe.32000.en-de \
  --val_file_pattern test_data/newstest2013.tok.bpe.32000.en-de \
  --use_token_batch True \
  --batch_size 20 \
  --sort_type none \
  --pool_size 2000 \
  --shuffle False \
  --shuffle_batch False \
  --device GPU \
  --pass_num 1 \
  --use_token_batch False \
  --check_acc True > local.log 2>&1 &
