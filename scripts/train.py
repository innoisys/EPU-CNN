from models.epu import EPUNet
from models.subnetworks import Subnet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", features_num=4,
             subnet=Subnet, fc_hidden_units=512, classes=1)
epu.set_name("example-model")
es = EarlyStopping(monitor="val_loss", patience=40, verbose=1, restore_best_weights=True)

learning_rate = 0.01
momentum = 0.9
lr_decay = 1e-6
lr_drop = 150

reduce_lr = LearningRateScheduler(lambda epoch: learning_rate * (0.5 ** (epoch // lr_drop)))
optimizer = SGD(learning_rate=learning_rate, decay=lr_decay, momentum=momentum, nesterov=True)

epu.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=False)
epu.fit(x=EPUNet.get_pfm([], 128, 128), epochs=100, validation_data=[],
        batch_size=32,
        callbacks=[es, reduce_lr])

epu.save_weights("./trained_models/{}.h5".format(epu.get_name()))