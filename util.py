import numpy as np, theano, theano.tensor as T, functools
from collections import OrderedDict as ordict

def batches(dataset, batch_size=100):
    indices = list(range(len(dataset["features"])))
    np.random.shuffle(indices)
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset:offset + batch_size]
        yield ordict((source, value[batch_indices]) for source, value in dataset.items())

def function(inputs=(), outputs=(), updates=()):
    fn = theano.function(list(inputs), list(outputs), updates=list(updates))
    def call(fn, **input_values):
        return ordict(zip(outputs, fn(**input_values)))
    return functools.partial(call, fn)

def print_monitor(items, prefix=None):
    prefix = prefix + "_" if prefix else ""
    keywidth = max(len(variable.name) for variable in items.keys())
    for variable, value in items.items():
        print "    %s%-*s:%20.8f" % (prefix, keywidth, variable.name, value)

def append_log(logs, monitor, prefix=None):
    prefix = prefix + "_" if prefix else ""
    for k, v in monitor.items():
        logs.setdefault("%s%s" % (prefix, k.name), []).append(v)

def rmsprop(parameter, step, learning_rate=1e-3, decay_rate=0.9, max_scaling=1e5):
    prev_mss = theano.shared(value=parameter.get_value() * 0)
    mss = ((1 - decay_rate) * T.sqr(step)
           + decay_rate * prev_mss)
    step *= learning_rate / T.maximum(T.sqrt(mss), 1. / max_scaling)
    updates = [(prev_mss, mss)]
    return step, updates

def cast_floatX(fn):
    return lambda *args: fn(*args).astype(theano.config.floatX)

@cast_floatX
def glorot(shape):
    d = np.sqrt(6. / sum(shape))
    return np.random.uniform(-d, +d, size=shape)

@cast_floatX
def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]]
