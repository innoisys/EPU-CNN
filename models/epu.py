import cv2 as cv
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from copy import deepcopy
from utils.image_preprocessing import PreProcess, normalize, coarse_fine_representation, light_dark_representations, \
    lab_representation
from skimage.measure.entropy import shannon_entropy
from skimage.filters.thresholding import threshold_yen
from subnetworks import Subnet
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

    def refine_prm(self, prm: dict, height=256, width=256, apply_color_map=True) -> dict:
        refined_prms = {}
        for key, value in prm.items():
            temp_prm = (PreProcess.normalize(
                cv.resize(value, (width, height), interpolation=cv.INTER_CUBIC)) * 255).astype(
                np.uint8)

            thresh = threshold_yen(temp_prm)
            temp_prm = (temp_prm > thresh) * temp_prm
            if apply_color_map:
                temp_prm = cv.applyColorMap(temp_prm, cv.COLORMAP_HOT)
            refined_prms[key] = temp_prm
        return refined_prms

    def get_prm(self):
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

    @staticmethod
    def get_pfms(images: list, height: int, width: int):
        # Color PFMs
        red_green, blue_yellow = [], []
        # Texture PFMs
        light_dark, coarse_fine = [], []

        for image in tqdm(images):
            image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

            if type(image) is not np.ndarray:
                image = image.numpy()

            _, red_green_rep, blue_yellow_rep = lab_representation(image=image)
            red_green.append(normalize(red_green_rep, height, width))
            red_green.append(normalize(blue_yellow_rep, height, width))

            coarse_fine_rep = coarse_fine_representation(image=image)
            coarse_fine.append(normalize(coarse_fine_rep, height, width))

            light_dark_rep = light_dark_representations(image=image, sigma=3)
            light_dark.append(normalize(light_dark_rep, height, width))

        # Formatting Routine
        red_green = np.asarray(red_green).reshape(len(red_green), height, width, 1)
        blue_yellow = np.asarray(blue_yellow).reshape(len(blue_yellow), height, width, 1)
        coarse_fine = np.asarray(coarse_fine).reshape(len(coarse_fine), height, width, 1)
        light_dark = np.asarray(light_dark).reshape(len(light_dark), height, width, 1)

        return red_green, blue_yellow, coarse_fine, light_dark
