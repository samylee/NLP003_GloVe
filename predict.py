import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random


class VectorEvaluation(object):
    def __init__(self, vector_file_path):
        if os.path.exists(vector_file_path):
            self.vector_file_path = vector_file_path
        else:
            raise FileExistsError("file is not exists!")
        self.read_data()

    def _read_line(self, word, *vector):
        return word, np.asarray(vector, dtype=np.float32)

    def read_data(self):
        words = []
        vector = []
        with open(self.vector_file_path, "r", encoding="utf-8") as f:
            for line in f:
                word, vec = self._read_line(*line.split(" "))
                words.append(word)
                vector.append(vec)
        assert len(vector) == len(words)
        self.vector = np.vstack(tuple(vector))
        self.vocab = {w: i for i, w in enumerate(words)}
        self.idex2word = {i: w for w, i in self.vocab.items()}

    def get_similar_words(self, word, w_num=10):
        w_num = min(len(self.vocab), w_num)
        idx = self.vocab.get(word, None)
        if not idx:
            idx = random.choice(range(self.vector.shape[0]))
        result = cosine_similarity(self.vector[idx].reshape(1, -1), self.vector)
        result = np.array(result).reshape(len(self.vocab), )
        idxs = np.argsort(result)[::-1][:w_num]
        print(">>>" * 7)
        print(self.idex2word[idx])
        for i in idxs:
            print("%s : %.3f" % (self.idex2word[i], result[i]))
        print("<<<" * 7)


def predict():
    save_vector_file_name = "data/glove.txt"
    vec_eval = VectorEvaluation(save_vector_file_name)
    vec_eval.get_similar_words("加拿大")
    vec_eval.get_similar_words("男人")


if __name__ == "__main__":
    predict()