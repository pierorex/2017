"""
In this script we apply Transfer Learning to use the pre-trained VGG19 model
as a basis for a new model trained on Xing user-item-interaction data
"""

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import applications
from keras.models import Model
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import cv2


def ratio_multiplier(y):
    # ratio function used for downsampling the classes by it's respective multipliers as percentage
    multiplier = {0: 0.001, 1: 0.005, 2: 0.1, 3: 0.3, 4: 0.03, 5: 1}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * multiplier[key])
    return target_stats


print('load array for training')
save_x_file = open('data.npy', 'rb')
x = np.load(save_x_file)

save_y_file = open('target.npy', 'rb')
y = np.load(save_y_file)

print('resample the data to get more balanced classes')
print(sorted(Counter(y).items()))
x, y = RandomUnderSampler(random_state=42, ratio=ratio_multiplier).fit_sample(x, y)
print(sorted(Counter(y).items()))
x, y = RandomOverSampler(random_state=42).fit_sample(x, y)
print(sorted(Counter(y).items()))

print('one-hot encode the labels')
y = to_categorical(y)
nb_classes = len(y[0])
print('number of classes to predict: {}'.format(nb_classes))

print('resize data to fit VGG19 input')
print('original x.shape = {}'.format(x.shape))
x = np.array([cv2.resize(sample, dsize=(48, 48), interpolation=cv2.INTER_CUBIC) for sample in x])
print(x.shape)
x = np.array([np.dstack((sample, sample, sample)) for sample in x])
print('final x.shape = {}'.format(x.shape))

print('build train, test and validation sets')
new_x, x_test, new_y, y_test = train_test_split(
    x,
    y,
    test_size=0.2
)
x_train, x_val, y_train, y_val = train_test_split(
    new_x,
    new_y,
    test_size=0.15
)

print('DEFINE TRANSFER LEARNING MODEL')
model = applications.VGG19(weights="imagenet", include_top=False, input_shape=x_train[0].shape)

print('layers in pre-trained model: {}'.format(len(model.layers)))
# freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers:
    layer.trainable = False

model.summary()

# add custom Layers
m = model.output
m = Flatten()(m)
m = Dense(32, activation='relu')(m)
m = Dropout(0.5)(m)
m = Dense(32, activation='relu')(m)
output_layer = Dense(nb_classes, activation='softmax')(m)

# create the final model
final_model = Model(input=model.input, output=output_layer)

# compile the model
final_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# add a callback to stop if we start overfitting and another one to save the best model
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=1, verbose=1, mode='auto'),
    ModelCheckpoint('xing_vgg19.h5', monitor='val_acc', verbose=1, save_best_only=True,
                    save_weights_only=False, mode='auto', period=1)
]

# train
history = final_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=callbacks
)

# plot history
# history_dict = history.history
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, label='training loss')
# plt.plot(epochs, val_loss, label='validation loss')
# plt.plot(epochs, acc, label='training acc')
# plt.plot(epochs, val_acc, label='validation acc')
# plt.legend()
# plt.show()

# predict some instances
print(final_model.predict(x_test[:10]))

# evaluate
final_model.evaluate(x_test, y_test)
