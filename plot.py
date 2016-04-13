import os, sys, cPickle, zipfile
from collections import OrderedDict
from itertools import starmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from blocks.serialization import load
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

channel_names = sys.argv[1].split()
paths = sys.argv[2:]

def load_instance(label, path):
    with open(path, "rb") as file:
        log = load(file)
    return dict(label=label, path=path, log=log)

instances = list(starmap(load_instance, enumerate(paths)))
colors = cm.rainbow(np.linspace(0, 1, len(instances)))

for channel_name in channel_names:
    plt.figure()
    for color, instance in zip(colors, instances):
        try:
            plt.plot(instance["log"][channel_name],
                     label=instance["label"], c=color)
            #plt.yscale("log")
        except KeyError:
            logger.warning("%s does not have %s" % (instance["path"], channel_name))
    plt.legend()
    plt.title(channel_name)
plt.show()
import pdb; pdb.set_trace()
