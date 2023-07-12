import cv2 as cv
import numpy as np
import tensorflow as tf

from copy import deepcopy
from utils.image_preprocessing import PreProcess
from skimage.measure.entropy import shannon_entropy
from skimage.filters.thresholding import threshold_yen
from subnet import Subnet, SubnetTwo, SubnetEffNet, MiniVGGIncRes, DummySubnet, BinaryTask, MultiTask, DummyCapsule
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D, Flatten, Dense, concatenate, AveragePooling2D, \
    Softmax


class EPUNet(tf.keras.Model):

    def __init__(self, init_size, subnet_act="tanh", epu_act="sigmoid", subnet=Subnet, features_num=5, classes=1,
                 fc_hidden_units=512):
        super(EPUNet, self).__init__()
        self.classes = classes
        self.sub_nets = []
        self.features_num = features_num

        self._name = "epu_net"
        self._predicted_label = None

        # bias
        self.b = tf.Variable(tf.zeros([classes]), trainable=True)
        self.__interpret_output = None

        # Initialization of feature networks
        for batch in range(self.features_num):
            self.sub_nets.append(subnet(init_size, act=subnet_act,
                                        fc_hidden_units=fc_hidden_units,
                                        classes=classes))

        self.activate = Activation(epu_act)

    def call(self, x, **kwargs):
        _outputs = []

        for i, feature in enumerate(x):
            _outputs.append(self.sub_nets[i](feature))

        summation = tf.reduce_sum(_outputs, 0) + self.b

        self.__interpret_output = _outputs
        _output = self.activate(summation)
        self._predicted_label = _output

        return _output

    def get_subnets(self):
        return self.sub_nets

    def get_bias(self):
        return self.b.numpy()

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def get_predicted_label(self) -> np.ndarray:
        return self._predicted_label.numpy()

    def get_interpret_output(self) -> np.ndarray:
        return np.asarray(list(self.__interpret_output)).reshape(self.features_num, self.classes)

    def refine_pfm(self, pfm: dict, height=256, width=256, apply_color_map=True) -> dict:
        refined_pfms = {}
        for key, value in pfm.items():
            temp_pfm = (PreProcess.normalize(
                cv.resize(value, (width, height), interpolation=cv.INTER_CUBIC)) * 255).astype(
                np.uint8)

            thresh = threshold_yen(temp_pfm)
            temp_pfm = (temp_pfm > thresh) * temp_pfm
            if apply_color_map:
                temp_pfm = cv.applyColorMap(temp_pfm, cv.COLORMAP_HOT)
            refined_pfms[key] = temp_pfm
        return refined_pfms

    def get_pfm(self):
        feature_maps = {}
        for i, subnet in enumerate(self.sub_nets):
            entropies = []
            _raw_feature_maps = subnet.get_feature_maps().numpy()
            _, height, width, depth = _raw_feature_maps.shape
            for k in range(depth):
                entropies.append(shannon_entropy(PreProcess.min_max_norm(_raw_feature_maps[0, :, :, k])))
            sorted_entropies = deepcopy(entropies)
            sorted_entropies.sort(reverse=True)
            sorted_entropies = sorted_entropies[:len(sorted_entropies) // 2]

            feature_maps[i] = np.asarray(
                [PreProcess.min_max_norm(_raw_feature_maps[0, :, :, entropies.index(sorted_entropy)]) * 255 for
                 sorted_entropy in sorted_entropies]).mean(axis=0).astype(np.uint8)

        return feature_maps

    def get_statistics(self, image, gt_label, features=None, datasets_human_labels=None) -> np.ndarray:
        pass

    def get_dataset_statistics(self, dataset):
        pass