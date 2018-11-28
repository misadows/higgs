from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
import numpy as np
from sklearn.metrics import roc_auc_score
import os

class BoostedTree(object):
    def __init__(self, treedepth=5):
        self.model = XGBClassifier(max_depth=treedepth, silent=False, verbose_eval=True)

    def train(self, dataset):
        n = len(dataset.x)
        y = np.concatenate([np.ones(n), np.zeros(n)])
        self.model.fit(np.concatenate([dataset.x, dataset.x]), y, sample_weight=np.concatenate([dataset.wa, dataset.wb]))

    def test(self, dataset):
        n = len(dataset.x)
        y = np.concatenate([np.ones(n), np.zeros(n)])
        predictions = self.model.predict(dataset.x)
        # predictions = [round(value) for value in self.model.predict(dataset.x)]
        accuracy = accuracy_score(y, np.concatenate([predictions, predictions]), sample_weight=np.concatenate([dataset.wa, dataset.wb]))
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        auc = roc_auc_score(y, np.concatenate([predictions, predictions]), sample_weight=np.concatenate([dataset.wa, dataset.wb]))
        print("AUC: %.2f%%" % (auc * 100.0))

def run(args):
    data_path = args.IN

    print "Loading data"
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w_a = read_np(os.path.join(data_path, "rhorho_raw.w_a.npy"))
    w_b = read_np(os.path.join(data_path, "rhorho_raw.w_b.npy"))
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    print "Processing data"
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, w_a, w_b, perm)

    num_features = points.train.x.shape[1]
    print "Generated %d features" % num_features

    model = BoostedTree(treedepth=args.TREEDEPTH)
    model.train(points.train)
    model.test(points.valid)

def start(args):
    run(args)

if __name__ == "__main__":
    start(args = {})