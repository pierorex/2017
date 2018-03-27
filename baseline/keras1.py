'''
Baseline solution for the ACM Recsys Challenge 2017
using XGBoost

by Daniel Kohlsdorf
'''

import numpy as np
import multiprocessing

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from model import *
from parser import *
from recommendation_worker import *

print(" --- Recsys Challenge 2017 Baseline --- ")

N_WORKERS         = 4
DATA_FOLDER       = '../../xing/data/recsys/sample_big'
# DATA_FOLDER       = '../../data'
USERS_FILE        = "{}/users.csv".format(DATA_FOLDER)
ITEMS_FILE        = "{}/items.csv".format(DATA_FOLDER)
INTERACTIONS_FILE = "{}/interactions.csv".format(DATA_FOLDER)
TARGET_USERS      = "{}/targetUsers.csv".format(DATA_FOLDER)
TARGET_ITEMS      = "{}/targetItems.csv".format(DATA_FOLDER)


'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
(header_users, users) = select(USERS_FILE, lambda x: True, build_user, lambda x: int(x[0]))
(header_items, items) = select(ITEMS_FILE, lambda x: True, build_item, lambda x: int(x[0]))

builder = InteractionBuilder(users, items)
(header_interactions, interactions) = select(
    INTERACTIONS_FILE,
    lambda x: x[2] != '0',
    builder.build_interaction,
    lambda x: (int(x[0]), int(x[1]))
)


# built interactions are only those that make sense, e.g. user exists and item exists

'''
2) Build recsys training data
'''
data    = np.array([interactions[key].features() for key in interactions.keys()])
labels  = np.array([interactions[key].label() for key in interactions.keys()])

print("saving")
merged_data = [interactions[key] for key in interactions.keys()]
Interaction.save(merged_data, 'merged_dataset.csv')
print("saved!")
exit(0)

# print('DATA')
# print(data, type(data))
# print('end')
# print('LABELS')
# print(labels.shape, type(labels))
# print('end')
# dataset = xgb.DMatrix(data, label=labels)
# dataset.save_binary("recsys2017.buffer")

"""
Create F1 metric
"""
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score


class MetricsCallback(Callback):
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        print(self.model.__dict__.keys())
        print(self.model.model.__dict__.keys())
        val_predict = (np.asarray(self.model.predict(self.model.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' - val_f1: {} - val_precision: {} - val_recall: {}'.format(_val_f1, _val_precision, _val_recall))


metrics_callback = MetricsCallback()


"""
calculate score to create targets (y)
"""


def premium_boost(user):
    return 2 if (user.isPremium) else 1


def user_success(item, user):
    (1 if clicked else 0 + 5 if (bookmarked or replied) else 0 + 20 if recruiter_interest else 0 - 10 if deleted else 0) * premium_boost(user)


def item_success(item, users):
    if users.filter(lambda u: user_success(item, u) > 0).size >= 1:
        return 50 if item.isPaid else 25
    else:
        return 0


def score(item, users):
    sum(users.map(lambda u: user_success(item, u))) + item_success(item, users)




"""
Build ML model
"""
input_dim = data.shape[1]
model = Sequential()

model.add(Dense(32, activation='relu', input_dim=input_dim))
# model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.5 ))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
#
# test_splitter = StratifiedShuffleSplit(1, test_size=0.2)
# train_index, test_index  = test_splitter.split(data, labels)
#
# x_train = data[]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128, callbacks=[metrics_callback])

sample = np.expand_dims(data[0], axis=0)
print(model.predict(sample))
bst = model  # to follow the same pattern as before

'''
3) Train XGBoost regression model with maximum tree depth of 2 and 25 trees
'''


'''
4) Create target sets for items and users
'''
print('creating target sets')
target_users = []
for n, line in enumerate(open(TARGET_USERS)):
    # there is a header in target_users in dataset
    if n == 0:
         continue
    target_users += [int(line.strip())]
target_users = set(target_users)

target_items = []
for line in open(TARGET_ITEMS):
    target_items += [int(line.strip())]


'''
5) Schedule classification
'''
print('testing with targetUsers and targetItems')
filename = "solution_" + str(0) + ".csv"
classify_worker(target_items, target_users, items, users, filename, bst)
