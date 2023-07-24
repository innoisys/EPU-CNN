import glob
import cv2 as cv
import numpy as np

from models.epu import EPUNet
from models.subnetworks import Subnet

test = [cv.imread(img) for img in glob.glob("../data/banapple/test/*.jpg")]

# pfm_order is useful to keep track of the order of the pfms and their respective subnetworks
# it used later in the inference to get the corresponding PRMs and RSSs in the correct order
# the following order is the default if it is not passed as an argument to EPU-NET
pfm_order = ["green-red", "blue-yellow", "coarse-fine", "light-dark"]
epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", features_num=4,
             subnet=Subnet, fc_hidden_units=512, classes=1, pfm_order=pfm_order)

epu.set_name("example-model")
epu.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=False)

epu.initialize_model(np.zeros((1, 128, 128, 1)))
epu.set_weights(np.load("../trained_models/{}.npy".format(epu.get_name()), allow_pickle=True))

img_idx = 5
output = epu(EPUNet.get_pfms(test[img_idx], 128, 128))

interpretation = epu.get_interpret_output()

prm_interpretations = epu.refine_prm(epu.get_prm())
overlayed_prms = epu.overlay_prms(test[img_idx], prm_interpretations)

stacked_prms = np.hstack(list(overlayed_prms.values()))
cv.imshow("Overlayed PRMs", stacked_prms)
cv.waitKey(0)
cv.destroyAllWindows()

epu.plot_rss()
print("RSSs: Green-Red {} || Blue-Yellow {} || "
      "Coarse-Fine {} || Light-Dark {}".format(interpretation[0], interpretation[1],
                                               interpretation[2], interpretation[3]))

