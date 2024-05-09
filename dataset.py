import os
import numpy as np
from scipy.sparse import lil_matrix
from collections import Counter
from torch.utils.data import Dataset


class CorpusPreprocess(object):
    def __init__(self, file_path, min_freq):
        self.file_path = file_path
        self.min_freq = min_freq
        self.huffman = None
        self.huffman_left = None
        self.huffman_right = None
        self.vocab = Counter()
        self.cooccurrence_matrix = None
        self.idex2word = None
        self.word2idex = None
        self.nag_sampling_vocab = None
        self._build_vocab()

    def _read_data(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"file path {self.file_path} is not exist !")
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.strip().split(" ")

    def _build_vocab(self):
        for line in self._read_data():
            self.vocab.update(line)
        self.vocab = dict((w.strip(), f) for w, f in self.vocab.items() if (f >= self.min_freq and w.strip()))
        self.vocab = {w: (i, f) for i, (w, f) in enumerate(self.vocab.items())}
        self.idex2word = {i: w for w, (i, f) in self.vocab.items()}

    def _build_cooccurrence_matrix(self, windows_size=5):
        if not self.vocab:
            self._build_vocab()
        self.cooccurrence_matrix = lil_matrix((len(self.vocab), len(self.vocab)), dtype=np.float32)
        for line in self._read_data():
            sentence_length = len(line)
            for i in range(sentence_length):
                center_w = line[i]
                if center_w not in self.vocab:
                    continue
                left_ws = line[max(i - windows_size, 0):i]
                for i, w in enumerate(left_ws[::-1]):
                    if w not in self.vocab:
                        continue
                    self.cooccurrence_matrix[self.vocab[center_w][0], self.vocab[w][0]] += 1.0 / (i + 1.0)
                    # right_ws not used, cause cooccurrence_matrix is Symmetric Matrices
                    self.cooccurrence_matrix[self.vocab[w][0], self.vocab[center_w][0]] += 1.0 / (i + 1.0)

    def get_cooccurrence_matrix(self, windows_size):
        if self.cooccurrence_matrix == None:
            self._build_cooccurrence_matrix(windows_size)
        return self.cooccurrence_matrix

    def get_vocab(self):
        if not isinstance(self.vocab, dict):
            self._build_vocab()
        return self.vocab


class TrainData(Dataset):
    def __init__(self, coo_matrix):
        self.coo_matrix = [((i, j), coo_matrix.data[i][pos]) for i, row in enumerate(coo_matrix.rows) for pos, j in enumerate(row)]
        self.x_max = 100
        self.alpha = 0.75

    def __len__(self):
        return len(self.coo_matrix)

    def __getitem__(self, idex):
        sample_data = self.coo_matrix[idex]
        sample = {
            "c": sample_data[0][0],
            "s": sample_data[0][1],
            "X_c_s": sample_data[1],
            "W_c_s": self.fw(sample_data[1])
        }
        return sample

    def fw(self, X_c_s):
        return (X_c_s / self.x_max) ** self.alpha if X_c_s < self.x_max else 1