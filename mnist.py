import cPickle, gzip, os

def get_data():
    data = cPickle.load(gzip.open(os.environ["MNIST_PKL_GZ"], 'rb'))
    which_sets = "train valid test".split()
    return dict((which_set, dict(features=x.astype("float32"),
                                 targets=y.astype("int32")))
                for which_set, (x, y) in zip(which_sets, data))
