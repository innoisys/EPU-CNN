import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, MaxPool2D, Dropout, Flatten, concatenate, Add
from abc import ABC, abstractmethod


class SubnetABC(ABC, tf.keras.Model):

    @abstractmethod
    def build_graph(self, input_shape):
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def get_feature_maps(self):
        raise NotImplementedError("Subclass must implement abstract method")


class ConvBNRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), strides=1, padding='same',
                 weight_decay=0.0005, rate=0.4, drop=True):

        super(ConvBNRelu, self).__init__()
        self.drop = drop
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding=padding, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.batchnorm = tf.keras.layers.BatchNormalization()

        if self.drop:
            self.dropOut = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, **kwargs):
        layer = self.conv(x)
        layer = self.batchnorm(layer)
        layer = tf.nn.relu(layer)
        if self.drop:
            layer = self.dropOut(layer)

        return layer


class SubnetTwo(SubnetABC):
    def __init__(self, init_size, fc_hidden_units, act="softmax", classes=10):
        super(SubnetTwo, self).__init__()
        self.act = act
        self.conv_one_a = ConvBNRelu(init_size, kernel_size=(3, 3), padding="same", rate=0.3)
        self.conv_one_b = ConvBNRelu(init_size, kernel_size=(3, 3), padding="same", drop=False)

        self.conv_two_a = ConvBNRelu(init_size * 2, (3, 3), padding="same")
        self.conv_two_b = ConvBNRelu(init_size * 2, (3, 3), padding="same", drop=False)

        self.conv_three_a = ConvBNRelu(init_size * 4, (3, 3), padding="same")
        self.conv_three_b = ConvBNRelu(init_size * 4, (3, 3), padding="same")
        self.conv_three_c = ConvBNRelu(init_size * 4, (3, 3), padding="same", drop=False)

        self.dense_one = Dense(fc_hidden_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.bn_af = BatchNormalization(momentum=0.99, scale=True, trainable=True)
        self.dropout = Dropout(rate=0.8)
        self.last_dense = Dense(classes, activation=act)
        self._feature_maps = None

    def call(self, x, *args, **kwargs):
        x = self.conv_one_a(x)
        x = self.conv_one_b(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = self.conv_two_a(x)
        self._feature_maps = self.conv_two_b(x)
        x = MaxPool2D(pool_size=(2, 2))(self._feature_maps)

        x = self.conv_three_a(x)
        x = self.conv_three_b(x)
        x = self.conv_three_c(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = self.dense_one(x)
        x = self.bn_af(x)
        x = self.dropout(x)

        output = self.last_dense(x)

        return output

    def build_graph(self, input_shape):
        input = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[input], outputs=self.call(input))

    def get_feature_maps(self):
        return self._feature_maps


class SubnetAVG(SubnetABC):

    def __init__(self, init_size, act="softmax",
                 classes=10, *args, **kwargs):
        super(SubnetAVG, self).__init__()
        self._name = "SubnetAVG"
        self.act = act
        self.conv_one_a = Conv2D(init_size, (3, 3), padding="same", activation="relu")
        self.conv_one_b = Conv2D(init_size, (3, 3), padding="same", activation="relu")
        self.max_pooling_one = MaxPool2D(pool_size=(2, 2))
        self.bn_b1 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.conv_two_a = Conv2D(init_size * 2, (3, 3), padding="same", activation="relu")
        self.conv_two_b = Conv2D(init_size * 2, (3, 3), padding="same", activation="relu")
        self.max_pooling_two = MaxPool2D(pool_size=(2, 2))
        self.bn_b2 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.conv_three_a = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.conv_three_b = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.conv_three_c = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.max_pooling_three = MaxPool2D(pool_size=(2, 2))
        self.bn_b3 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.last_dense = Dense(classes, activation=act)
        self.flatten = Flatten()
        self._feature_maps = None

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def call(self, x, **kwargs):  # implement forward pass
        x = self.conv_one_a(x)
        x = self.conv_one_b(x)
        x = self.max_pooling_one(x)
        x = self.bn_b1(x)

        x = self.conv_two_a(x)
        self._feature_maps = self.conv_two_b(x)
        x = self.max_pooling_two(self._feature_maps)
        x = self.bn_b2(x)

        x = self.conv_three_a(x)
        x = self.conv_three_b(x)
        x = self.conv_three_c(x)
        x = self.max_pooling_three(x)
        x = self.bn_b3(x)
        x = self.gap(x)

        out_x = self.flatten(x)
        output = self.last_dense(out_x)
        return output

    def get_feature_maps(self):
        return self._feature_maps

    def get_name(self):
        return self._name

    def set_name(self, name: str):
        self._name = name


class Subnet(tf.keras.Model):
    def __init__(self, init_size, act="softmax", classes=10, fc_hidden_units=512, *args, **kwargs):
        super(Subnet, self).__init__()
        self.act = act
        self.conv_one_a = Conv2D(init_size, (3, 3), padding="same", activation="relu")
        self.conv_one_b = Conv2D(init_size, (3, 3), padding="same", activation="relu")
        self.bn_b1 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.conv_two_a = Conv2D(init_size * 2, (3, 3), padding="same", activation="relu")
        self.conv_two_b = Conv2D(init_size * 2, (3, 3), padding="same", activation="relu")
        self.bn_b2 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.conv_three_a = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.conv_three_b = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.conv_three_c = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.bn_b3 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.dense_one = Dense(fc_hidden_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.dense_two = Dense(fc_hidden_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.last_dense = Dense(classes, activation=act)

        self._feature_maps = None

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def call(self, x, **kwargs):  # implement forward pass
        x = self.conv_one_a(x)
        x = self.conv_one_b(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = self.bn_b1(x)

        x = self.conv_two_a(x)
        self._feature_maps = self.conv_two_b(x)
        x = MaxPool2D(pool_size=(2, 2))(self._feature_maps)
        x = self.bn_b2(x)

        x = self.conv_three_a(x)
        x = self.conv_three_b(x)
        x = self.conv_three_c(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = self.bn_b3(x)

        x = Flatten()(x)
        x = self.dense_one(x)
        x = self.dense_two(x)

        output = self.last_dense(x)
        return output

    def get_feature_maps(self):
        return self._feature_maps


class Mini_VGG(tf.keras.Model):
    def __init__(self, init_size, act, classes=1, feature_name="base", *args, **kwargs):
        super(Mini_VGG, self).__init__()
        self.act = act

        self.conv_one_a = Conv2D(init_size, (3, 3), padding="same", activation="relu")
        self.conv_one_b = Conv2D(init_size, (3, 3), padding="same", activation="relu")
        self.bn_b1 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.conv_two_a = Conv2D(init_size * 2, (3, 3), padding="same", activation="relu")
        self.conv_two_b = Conv2D(init_size * 2, (3, 3), padding="same", activation="relu")
        self.bn_b2 = BatchNormalization(momentum=0.9, scale=True, trainable=True)

        self.conv_three_a = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.conv_three_b = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.conv_three_c = Conv2D(init_size * 4, (3, 3), padding="same", activation="relu")
        self.bn_b3 = BatchNormalization(momentum=0.9, scale=True, trainable=True)  # , fused=True

        self.dense_one = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.bn_af = BatchNormalization(momentum=0.99, scale=True, trainable=True)  # after flatten
        self.last_dense = Dense(classes, activation=act)
        self.out_batch_one = None

        self.feature_maps = None

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def call(self, x, **kwargs):
        x = self.conv_one_a(x)
        x = self.conv_one_b(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = self.bn_b1(x)

        x = self.conv_two_a(x)
        self.feature_maps = self.conv_two_b(x)
        x = MaxPool2D(pool_size=(2, 2))(self.feature_maps)

        x = self.bn_b2(x)
        x = self.conv_three_a(x)
        x = self.conv_three_b(x)
        x = self.conv_three_c(x)

        x = MaxPool2D(pool_size=(2, 2))(x)
        x = self.bn_b3(x)

        x = Flatten()(x)

        x = self.dense_one(x)
        x = Dropout(0.5)(x)

        output = self.last_dense(x)
        return output

    def get_feature_maps(self):
        return self.feature_maps
