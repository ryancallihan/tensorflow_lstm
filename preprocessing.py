from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

class Indexer:
    def __init__(self):
        self.c2n = dict()
        self.n2c = list()
        self.start_idx = 1

    def fit_transform(self, value, add_if_new=True):
        n = self.c2n.get(value)

        if n is None and add_if_new:
            n = len(self.n2c) + self.start_idx
            self.c2n[value] = n
            self.n2c.append(value)
        return n

    def value(self, number):
        return self.n2c[number]

    def max_number(self):
        return len(self.n2c)

    def transform(self, data, matrix=False, add_if_new=True):
        vectors = []
        for line in data:
            if matrix:
                line_built = []
                for c in line:
                    line_built.append(self.fit_transform(c, add_if_new=True))
                vectors.append(line_built)
            else:
                vectors.append(self.fit_transform(line, add_if_new=True))
        return vectors

def generate_instances(
        data,
        labels_data,
        n_word,
        n_label,
        max_timesteps,
        batch_size=128):
    if labels_data == None:
        labels_data = np.zeros((len(data), ), dtype=np.int32)

    n_batches = len(data) // batch_size

    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            n_label),
        dtype=np.float32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    words = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            word = data[(batch * batch_size) + idx]

            label = labels_data[(batch * batch_size) + idx]
            labels[batch, idx, label] = 1

            # Sequence
            timesteps = min(max_timesteps, len(word))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Word characters
            words[batch, idx, :timesteps] = word[:timesteps]

    return (words, lengths, labels, n_word)


def read_file(filename, label=False, raw=False):
    with open(filename) as file:
        lines = file.readlines()
    if raw:
        return lines
    elif label:
        return [int(line.strip()) for line in lines]
    else:
        return [list(line.strip()) for line in lines]



# if __name__ == "__main__":

