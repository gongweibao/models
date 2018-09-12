import glob
import os
import random
import tarfile
import cPickle

import multiprocessing
import math
import copy
import logging
import functools
import paddle

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

def load_src_trg_ids(src_vocab, trg_vocab, config, fpattern, tar_fname, only_src=True):
    converters = [
        Converter(
            vocab=src_vocab,
            beg=src_vocab[config.start_mark],
            end=src_vocab[config.end_mark],
            unk=src_vocab[config.unk_mark],
            delimiter=config.token_delimiter)
    ]
    if not only_src:
        converters.append(
            Converter(
                vocab=trg_vocab,
                beg=trg_vocab[config.start_mark],
                end=trg_vocab[config.end_mark],
                unk=trg_vocab[config.unk_mark],
                delimiter=config.token_delimiter))

    converters = ComposedConverter(converters)

    src_ids=[]
    trg_ids=[]
    samples=[]

    total=0
    for i, line in enumerate(load_lines(fpattern, tar_fname, config.field_delimiter, only_src)):
        if len(line) <= 0:
            continue

        src_trg_ids = converters(line)

        lens = [len(src_trg_ids[0])]
        src_ids.append(src_trg_ids[0])
        if not only_src:
            lens.append(len(src_trg_ids[1]))
            trg_ids.append(src_trg_ids[1])

        samples.append([i, max(lens), min(lens)])

        if len(src_ids) >= 1000:
            total += len(src_ids)
            yield (src_ids, trg_ids, samples)
            src_ids=[]
            trg_ids=[]
            samples=[]

    total += len(src_ids)
    logging.debug("file:{} total_read:{}\n".format(fpattern, total))
    yield (src_ids, trg_ids, samples)

def load_lines(fpattern, tar_fname, field_delimiter, only_src=True):
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
            if (not only_src and len(fields) == 2) or (
                    only_src and len(fields) == 1):
                yield fields
    else:
        for fpath in fpaths:
            logging.debug("open file:{}".format(fpath))
            if not os.path.isfile(fpath):
                raise IOError("Invalid file: %s" % fpath)

            with open(fpath, "r") as f:
                for line in f:
                    fields = line.strip("\n").split(field_delimiter)
                    if (not only_src and len(fields) == 2) or (
                            only_src and len(fields) == 1):
                        yield fields
            logging.debug("finish file:{}".format(fpath))

def load_dict(dict_path, reverse=False):
    word_dict = {}
    with open(dict_path, "r") as fdict:
        for idx, line in enumerate(fdict):
            if reverse:
                word_dict[idx] = line.strip("\n")
            else:
                word_dict[line.strip("\n")] = idx
    return word_dict

def file_reader(src_vocab, trg_vocab, config, fpattern, tar_fname, only_src=True):
    reader = load_src_trg_ids(src_vocab, trg_vocab, config, fpattern, tar_fname, only_src);
    return reader

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

        self._src_vocab = load_dict(self._config.src_vocab_fpath)
        self._only_src = True
        if self._config.trg_vocab_fpath is not None:
            self._trg_vocab = load_dict(self._config.trg_vocab_fpath)
            self._only_src = False

        self._src_seq_ids = []
        self._trg_seq_ids = []

    def load_data(self):
        if isinstance(self._config.fpattern, list):
            fpaths = self._config.fpattern
        else:
            fpaths = glob.glob(self._config.fpattern)
        assert len(fpaths) > 0, "no input files"

        process_num = 5
        size = int(math.ceil(float(len(fpaths)) / process_num))
        readers=[]
        for i in range(process_num):
             f = fpaths[i * size:(i + 1) * size]
             readers.append(
                functools.partial(file_reader, self._src_vocab, 
                                  self._trg_vocab, self._config, f, self._config.tar_fname, self._only_src))

            
        for src_ids, trg_ids, samples in paddle.reader.multiprocess_reader(readers, queue_size=10000, use_pipe=True)():
            self._src_seq_ids += src_ids
            if not self._only_src:
                self._trg_seq_ids += trg_ids
            self._sample_infos += samples

        idx = 0
        for s in self._sample_infos:
            idx += 1
            s[0] = idx

        logging.debug("src_trg_ids len:{} trg_seq_ids len:{}".format(len(self._src_seq_ids), len(self._trg_seq_ids)))

    def get_sample_infos(self):
        return self._sample_infos

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



