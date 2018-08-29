import unittest
from reader import *
from config import *

class TestReader(unittest.TestCase):
    def test_multiprocess(self):
        config = ReaderConfig()

        config.src_vocab_fpath="./test_data/vocab.bpe.32000"
        config.trg_vocab_fpath="./test_data/vocab.bpe.32000"
        config.fpattern="./test_data/train.tok.clean.bpe.32000.en-de_*"
        #token_delimiter=args.token_delimiter,
        #use_token_batch=args.use_token_batch,
        config.batch_size=2048
        config.pool_size=10000
        config.sort_type="pool"
        config.shuffle=True
        config.shuffle_batch=True
        #start_mark=args.special_token[0],
        #end_mark=args.special_token[1],
        #unk_mark=args.special_token[2],
        # count start and end tokens out
        config.max_length=ModelHyperParams.max_length - 2,
        config.clip_last_batch=False

        reader = MultiProcessReader(config=config) 

    def test_singleproces(self):
        pass

if __name__ == "__main__":
    unittest.main()
