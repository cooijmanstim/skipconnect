import sys, logging, numpy as np, theano, theano.tensor as T
from collections import OrderedDict as ordict

import mnist, util
from blocks import serialization

logging.basicConfig()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sequence_length = 784
    nclasses = 10
    nhidden = 80
    batch_size = 100

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--skips", nargs="+", type=int, default=[])
    parser.add_argument("--instance-dependent", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    inputs = ordict(features=T.matrix("features"),
                    targets=T.ivector("targets"))
    x, y = inputs["features"], inputs["targets"]

    theano.config.compute_test_value = "warn"
    x.tag.test_value = np.random.random((11, 784)).astype(theano.config.floatX)
    y.tag.test_value = np.random.random_integers(low=0, high=9, size=(11,)).astype(np.int32)

    # move time axis before batch axis
    x = x.T

    # (time, batch, features)
    x = x.reshape((x.shape[0], x.shape[1], 1))

    Wx = theano.shared(util.orthogonal((1, nhidden)), name="Wx")
    bx = theano.shared(np.zeros((nhidden,), dtype=theano.config.floatX), name="bx")
    Wy = theano.shared(util.orthogonal((nhidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")
    Wh = theano.shared(0.9*np.eye(nhidden, dtype=theano.config.floatX), name="Wh")
    h0 = theano.shared(np.zeros((nhidden,), dtype=theano.config.floatX), name="h0")

    parameters = ordict((var.name, var) for var in [Wx, bx, Wy, by, Wh, h0])

    taps = [-1] + [-m for m in args.skips]
    if args.instance_dependent:
        Wskip = theano.shared(np.zeros((nhidden, len(taps),), dtype=theano.config.floatX), name="Wskip")
        bskip = theano.shared(np.zeros((len(taps),), dtype=theano.config.floatX), name="bskip")
        parameters["Wskip"] = Wskip
        parameters["bskip"] = bskip

        def compute_htilde(hs):
            energies = T.dot(hs[0], Wskip) + bskip
            ws = T.nnet.softmax(energies)
            return sum(hs[i] * ws[:, i, None] for i in range(len(hs)))
    else:
        s = theano.shared(np.zeros((len(taps),), dtype=theano.config.floatX), name="s")
        parameters["s"] = s
        ws = T.nnet.softmax(s).copy(name="ws")

        def compute_htilde(hs):
            return sum(hs[i] * ws[0, i] for i in range(len(hs)))

    xtilde = T.dot(x, Wx) + bx
    dummy_h = T.zeros((x.shape[0], x.shape[1], nhidden))
    def stepfn(x, dummy_h, *hs):
        assert len(hs) == len(taps)
        htilde = compute_htilde(hs)
        return dummy_h + T.tanh(T.dot(htilde, Wh) + x)
    # all taps have h0 as default to avoid strange edge effects
    initial_hs = T.repeat(T.repeat(h0[None, None, :],
                                   -min(taps), axis=0),
                          x.shape[1], axis=1)
    h, _ = theano.scan(stepfn,
                       sequences=[xtilde, dummy_h],
                       outputs_info=[dict(initial=initial_hs, taps=taps)])
    ytilde = T.dot(h[-1], Wy) + by
    yhat = T.nnet.softmax(ytilde)

    error_rate = T.neq(y, T.argmax(yhat, axis=1)).mean()
    cross_entropy = T.nnet.categorical_crossentropy(yhat, y).mean()

    cost = cross_entropy

    hidden_norms = []
    hidden_grad_norms = []
    if False:
        h_grads = T.grad(cross_entropy, dummy_h)
        h_grads_bnstats = T.stack([h_grads.mean(axis=1), h_grads.var(axis=1)])
        h_bnstats = T.stack([h.mean(axis=1), h.var(axis=1)])
        timesubsample = 100
        for t in xrange(0, sequence_length, timesubsample):
            #hidden_grad_norms.append(h_grads_bnstats[:, t, :].mean(axis=1) .copy(name="grad_bnstats_mean:h_%03i" % t))
            hidden_grad_norms.append(h_grads[t, :, :].norm(2).copy(name="grad_norm:h_%03i" % t))
            hidden_norms.append(h[t, :, :].norm(2).copy(name="norm:h_%03i" % t))
            #hidden_norms.append(h_bnstats[:, t, :].mean(axis=1) .copy(name="bnstats_mean:h_%03i" % t))

    steps = []
    steprule_updates = []
    for parameter, gradient in zip(parameters.values(), T.grad(cost, list(parameters.values()))):
        step, updates = util.rmsprop(parameter, gradient)
        steps.append((parameter, -step))
        steprule_updates.extend(updates)

    # step norm clipping
    total_step_norm = T.concatenate([step.flatten() for _, step in steps]).norm(2)
    max_step_norm = 1.0
    steps = ordict((parameter, T.switch(total_step_norm < max_step_norm,
                                        step, step / total_step_norm))
                   for parameter, step in steps)

    objectives = [locals()[name].copy(name=name) for name in
                  "cost cross_entropy error_rate".split()]
    train_fn   = util.function(list(inputs.values()),
                               ([step.norm(2).copy(name="step_norm:%s" % parameter.name)
                                 for parameter, step in steps.items()]
                                + objectives + hidden_norms + hidden_grad_norms),
                               updates=([(parameter, parameter + step)
                                         for parameter, step in steps.items()]
                                        + steprule_updates))
    monitor_fn = util.function(list(inputs.values()), objectives)
    datasets = mnist.get_data()

    patience = 50
    time_since_improvement = 0
    best_valid_error_rate = 1

    log = dict()
    for epoch in range(args.num_epochs):
        print "epoch", epoch
        for batch in util.batches(datasets["train"], batch_size=args.batch_size):
            sys.stdout.write(".")
            sys.stdout.flush()
            monitor = train_fn(**batch)
            util.append_log(log, monitor, prefix="iteration")
        print
        util.print_monitor(monitor, prefix="iteration")
        for which_set in "train valid test".split():
            monitor = monitor_fn(**next(util.batches(datasets[which_set], batch_size=1000)))
            util.append_log(log, monitor, prefix=which_set)
            util.print_monitor(monitor, prefix=which_set)
        serialization.secure_dump(log, "log.pkl")

        if log["valid_error_rate"][-1] < best_valid_error_rate:
            best_valid_error_rate = log["valid_error_rate"][-1]
            serialization.secure_dump(dict((parameter.name, parameter.get_value())
                                           for parameter in parameters.values()),
                                      "best_parameters.pkl")
            time_since_improvement = 0
        else:
            time_since_improvement += 1
        if time_since_improvement > patience:
            break
