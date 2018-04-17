from sklearn.model_selection import train_test_split
import numpy as np

print('load array for training')
save_x_file = open('data.npy', 'rb')
x = np.load(save_x_file)

save_y_file = open('target.npy', 'rb')
y = np.load(save_y_file)

print('resample the data to get more balanced classes')
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.datasets import make_imbalance
from collections import Counter

def ratio_multiplier(y):
    multiplier = {0: 0.001, 1: 0.005, 2: 0.1, 3: 0.3, 4: 0.03, 5: 1}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * multiplier[key])
    return target_stats

print(sorted(Counter(y).items()))
x, y = RandomUnderSampler(random_state=42, ratio=ratio_multiplier).fit_sample(x, y)
print(sorted(Counter(y).items()))
x, y = RandomOverSampler(random_state=42).fit_sample(x, y)
print(sorted(Counter(y).items()))

print('one-hot encode the labels')
from keras.utils import to_categorical

nb_classes = len(np.unique(y))
print('number of classes to predict: {}'.format(nb_classes))
y = to_categorical(y)

print('resize data to fit VGG19 input')
import cv2
print('original x.shape = {}'.format(x.shape))
x = cv2.merge((x,x,x))
x = np.array([cv2.resize(sample, dsize=(48, 48), interpolation=cv2.INTER_CUBIC) for sample in x])
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
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

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

# creat the final model
final_model = Model(input=model.input, output=output_layer)

# compile the model
final_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# add a callback to stop if we start overfitting and another one to save the best model
from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=1, verbose=1, mode='auto'),
    ModelCheckpoint('xing_vgg19.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
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
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.plot(epochs, acc, label='training acc')
plt.plot(epochs, val_acc, label='validation acc')
plt.legend()
plt.show()

# predict some instances
print(final_model.predict(x_test[:10]))

# evaluate
final_model.evaluate(x_test, y_test)
