#import unittest
from reader import *
from config import *
import sys

class TestReader(object):
    def test_multiprocess(self):
        config = ReaderConfig()

        config.src_vocab_fpath="./test_en_fr/vocab.wordpiece.en-fr"
        config.trg_vocab_fpath="./test_en_fr/vocab.wordpiece.en-fr"
        config.fpattern="./test_en_fr/train.wordpiece.en-fr_000_*"

        #config.src_vocab_fpath="./test_data/vocab.bpe.32000"
        #config.trg_vocab_fpath="./test_data/vocab.bpe.32000"
        #config.fpattern="./test_data/train.tok.clean.bpe.32000.en-de_*"

        token_delimiter="\x01"
        config.batch_size=2048
        config.pool_size=200000
        config.sort_type="pool"
        config.shuffle=True
        config.shuffle_batch=True
        config.max_length=ModelHyperParams.max_length - 2,
        config.clip_last_batch=False

        reader = DataReader(config=config) 
        reader.load_data()

    """
    def test_append(self):
        l = []
        for i in range(50000000):
            l.append([1,2,3])
    """

if __name__ == "__main__":
    import logging
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    #unittest.main()
    t = TestReader()
    t.test_multiprocess()
