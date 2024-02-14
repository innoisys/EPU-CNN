import glob
import cv2 as cv
import numpy as np

from models.epu import EPUNet
from models.subnetworks import Subnet
from tensorflow.keras.metrics import AUC

test = [cv.imread(img) for img in glob.glob("../data/banapple/test/*.jpg")]
labels = np.asarray([1 if "apple" in img.split("/")[-1] else 0 for img in glob.glob("../data/banapple/test/*.jpg")])

# pfm_order is useful to keep track of the order of the pfms and their respective subnetworks
# it used later in the inference to get the corresponding PRMs and RSSs in the correct order
# the following order is the default if it is not passed as an argument to EPU-NET
pfm_order = ["green-red", "blue-yellow", "coarse-fine", "light-dark"]
epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", features_num=4,
             subnet=Subnet, fc_hidden_units=512, classes=1, pfm_order=pfm_order)

epu.set_name("example-model")
epu.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy", AUC()], run_eagerly=False)

epu.initialize_model(np.zeros((1, 128, 128, 1)))
epu.set_weights(np.load("../trained_models/{}.npy".format(epu.get_name()), allow_pickle=True))
epu.evaluate(x=EPUNet.get_pfms(test, 128, 128), y=labels, batch_size=64)
