import unittest
from reader import *
from config import *

class TestReader(unittest.TestCase):
    def test_multiprocess(self):
        config = ReaderConfig()

        config.src_vocab_fpath="./test_en_fr/vocab.wordpiece.en-fr"
        config.trg_vocab_fpath="./test_en_fr/vocab.wordpiece.en-fr"
        config.fpattern="./test_en_fr/train.wordpiece.en-fr_*"
        token_delimiter="\x01"
        #use_token_batch=args.use_token_batch,
        config.batch_size=2048
        config.pool_size=200000
        config.sort_type="pool"
        config.shuffle=True
        config.shuffle_batch=True
        #start_mark=args.special_token[0],
        #end_mark=args.special_token[1],
        #unk_mark=args.special_token[2],
        # count start and end tokens out
        config.max_length=ModelHyperParams.max_length - 2,
        config.clip_last_batch=False

        reader = DataReader(config=config) 
        reader.load_data()

    def test_singleproces(self):
        pass

if __name__ == "__main__":
    unittest.main()
