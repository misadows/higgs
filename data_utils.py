import numpy as np
import random

class Dataset(object):
    def __init__(self, x, wa, wa02, wa04, wa06, wa08, wb):
        self.x = x[:, :-1]
        self.filt = x[:, -1]
        self.wa = wa
        self.wa02 = wa02
        self.wa04 = wa04
        self.wa06 = wa06
        self.wa08 = wa08

        self.wb = wb

        self.n = x.shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.wa = self.wa[perm]
        self.wa02 = self.wa02[perm]
        self.wa04 = self.wa04[perm]
        self.wa06 = self.wa06[perm]
        self.wa08 = self.wa08[perm]
        self.wb = self.wb[perm]
        self.filt = self.filt[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return (self.x[cur_id:cur_id+batch_size],
                self.wa[cur_id:cur_id+batch_size],
                self.wa02[cur_id:cur_id+batch_size],
                self.wa04[cur_id:cur_id+batch_size],
                self.wa06[cur_id:cur_id+batch_size],
                self.wa08[cur_id:cur_id+batch_size],
                self.wb[cur_id:cur_id+batch_size],
                self.filt[cur_id:cur_id+batch_size])


def read_np(filename):
    with open(filename) as f:
        return np.load(f)


class EventDatasets(object):

    def __init__(self, event, w_a, wa02, wa04, wa06, wa08, w_b, perm, filtered=False, raw=False, miniset=False,  unweighted=False):
        data = event.cols[:, :-1]
        filt = event.cols[:, -1]

        if miniset:
            print("Miniset")
            train_ids = perm[-300000:-200000]
            print(len(train_ids))
            valid_ids = perm[-200000:-100000]
            test_ids = perm[-100000:]
        else:
            train_ids = perm[:-200000]
            valid_ids = perm[-200000:-100000]
            test_ids = perm[-100000:]

        if not raw:
            print "SCALE!!"
            means = data[train_ids].mean(0)
            stds = data[train_ids].std(0)
            data = (data - means) / stds

        if filtered:
            train_ids = train_ids[filt[train_ids] == 1]
            valid_ids = valid_ids[filt[valid_ids] == 1]
            test_ids = test_ids[filt[test_ids] == 1]

        data = np.concatenate([data, filt.reshape([-1, 1])], 1)

        def unweight(x):
            return 0 if x < random.random()*2 else 1

        if unweighted:
            w_a = np.array(map(unweight, w_a))
            w_b = np.array(map(unweight, w_b))

        self.train = Dataset(data[train_ids], w_a[train_ids], wa02[train_ids], wa04[train_ids], wa06[train_ids], wa08[train_ids], w_b[train_ids])
        self.valid = Dataset(data[valid_ids], w_a[valid_ids],wa02[valid_ids], wa04[valid_ids], wa06[valid_ids], wa08[valid_ids], w_b[valid_ids])
        self.test = Dataset(data[test_ids], w_a[test_ids],wa02[test_ids], wa04[test_ids], wa06[test_ids], wa08[test_ids], w_b[test_ids])