import tensorflow as tf
from a1a1 import A1A1Event
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork
import os 


def run(args):
    data_path = args.IN

    print "Loading data"
    data = read_np(os.path.join(data_path, "a1a1_raw.data.npy"))
    w_a = read_np(os.path.join(data_path, "a1a1_raw.w_a.npy"))
    w_b = read_np(os.path.join(data_path, "a1a1_raw.w_b.npy"))
    perm = read_np(os.path.join(data_path, "a1a1_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    print "Processing data"
    event = A1A1Event(data, args)
    points = EventDatasets(event, w_a, w_b, perm, miniset=args.MINISET, unweighted=args.UNWEIGHTED)

    num_features = points.train.x.shape[1]
    print "Generated %d features" % num_features

    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print "Training"
    total_train(model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})