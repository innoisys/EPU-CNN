import cv2 as cv
import numpy as np

from scipy import ndimage as nd
from skimage.filters import sobel


class PreProcess(object):

    def __init__(self, size=(128, 128), preprocess="basic", interpolation=cv.INTER_CUBIC):
        """

        :param size:
        :param preprocess:
        :param interpolation:
        """
        self._width, self._height = size
        self._interpolation = interpolation

        if isinstance(preprocess, str):
            if preprocess is "basic":
                self._preprocess = PreProcess.basic_image_preprocess
        else:
            self._preprocess = preprocess

    def get_processed_image(self, image, generator=None, transform=None):

        if isinstance(image, str):
            image = cv.imread(image)
            image = self.resize(image)

            if generator is not None and transform is not None:
                image = generator.apply_transform(image, transform)

            return self.__get_preprocess(image)

        elif isinstance(image, list):
            return self.get_processed_image_list(image, generator=generator, transform=transform)
        else:
            image = self.resize(image)

            if generator is not None and transform is not None:
                image = generator.apply_transform(image, transform)

            return self.__get_preprocess(image)

    def get_processed_image_list(self, images, generator=None, transform=None):
        processed_images = []
        for image in images:
            processed_images.append(self.get_processed_image(image, generator=generator, transform=transform))
        return processed_images

    def resize(self, image):
        return cv.resize(image, (self._width, self._width),
                         interpolation=self._interpolation)

    def __get_preprocess(self, image):
        if isinstance(self._preprocess, list):
            features = []
            for preprocess in self._preprocess:
                features.append(preprocess(image))
            return features
        else:
            return self._preprocess(image)

    @staticmethod
    def min_max_norm(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def normalize(image):
        if type(image) is list:
            min_max_images = []
            for _image in image:
                min_max_images.append(PreProcess.normalize(_image))
            return min_max_images
        else:
            return PreProcess.min_max_norm(image)

    @staticmethod
    def basic_image_preprocess(image):
        return image.astype(np.float32) / 255.

    @staticmethod
    def numerical_sorting(list_item):
        if type(list_item) is not list:
            raise Exception("needs list object")
        list_item.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        return list_item


# EPU Tools

def normalize(image, height, width):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    return cv.resize(image, (height, width), interpolation=cv.INTER_CUBIC)


def lab_representation(image):
    lab_rep = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    return cv.split(lab_rep)


def coarse_fine_representation(image):
    return sobel(cv.cvtColor(image, cv.COLOR_BGR2GRAY))


def light_dark_representations(image, sigma=3):
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return nd.gaussian_filter(image_grayscale, sigma)
