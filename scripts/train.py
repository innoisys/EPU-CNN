import glob
import cv2 as cv
import numpy as np

from models.epu import EPUNet
from models.subnetworks import Subnet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

train = [cv.imread(img) for img in glob.glob("../data/banapple/train/*.jpg")]
train_labels = np.asarray([1 if "apple" in img else 0 for img in glob.glob("../data/banapple/train/*.jpg").split("/")[-1]],
                          dtype=np.float32)

validation = [cv.imread(img) for img in glob.glob("../data/banapple/validation/*.jpg")]
validation_labels = np.asarray([1 if "apple" in img.split("/")[-1] else 0 for img in glob.glob("../data/banapple/validation/*.jpg")],
                               dtype=np.float32)

# pfm_order is useful to keep track of the order of the pfms and their respective subnetworks
# it used later in the inference to get the corresponding PRMs and RSSs in the correct order
# the following order is the default if it is not passed as an argument to EPU-NET
pfm_order = ["green-red", "blue-yellow", "coarse-fine", "light-dark"]
epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", features_num=4,
             subnet=Subnet, fc_hidden_units=512, classes=1, pfm_order=pfm_order)
epu.set_name("example-model")
es = EarlyStopping(monitor="val_loss", patience=40, verbose=1, restore_best_weights=True)

learning_rate = 0.01
momentum = 0.9
lr_decay = 1e-6
lr_drop = 150

reduce_lr = LearningRateScheduler(lambda epoch: learning_rate * (0.5 ** (epoch // lr_drop)))
optimizer = SGD(learning_rate=learning_rate, decay=lr_decay, momentum=momentum, nesterov=True)

epu.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=False)
epu.fit(x=EPUNet.get_pfms(train, 128, 128), y=train_labels, epochs=1,
        validation_data=(EPUNet.get_pfms(validation, 128, 128), validation_labels),
        batch_size=1, callbacks=[reduce_lr, es])

np.save("../trained_models/{}.npy".format(epu.get_name()),
        np.array(epu.get_weights(), dtype=object), allow_pickle=True)
