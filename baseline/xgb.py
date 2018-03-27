'''
Baseline solution for the ACM Recsys Challenge 2017
using XGBoost

by Daniel Kohlsdorf
'''

import xgboost as xgb
import numpy as np
import multiprocessing

from keras.layers import Dense, Dropout
from keras.models import Sequential

from model import *
from parser import *
from recommendation_worker import *

print(" --- Recsys Challenge 2017 Baseline --- ")

N_WORKERS         = 4
DATA_FOLDER       = '../../xing/data/recsys/sample'
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
# print('DATA')
# print(data, type(data))
# print('end')
# print('LABELS')
# print(labels.shape, type(labels))
# print('end')
# dataset = xgb.DMatrix(data, label=labels)
# dataset.save_binary("recsys2017.buffer")

input_dim = data.shape[1]
model = Sequential()

model.add(Dense(units=32, activation='relu', input_dim=input_dim))
# model.add(Dropout(0.25))
model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(units=8, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(units=1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(data, labels, validation_split=0.2, epochs=2, batch_size=128)

test = np.expand_dims(data[0], axis=0)
print(model.predict(test))
bst = model  # to follow the same pattern as before

'''
3) Train XGBoost regression model with maximum tree depth of 2 and 25 trees
'''
# evallist = [(dataset, 'train')]
# param = {'bst:max_depth': 2, 'bst:eta': 0.1, 'silent': 1, 'objective': 'reg:linear' }
# param['nthread']     = 4
# param['eval_metric'] = 'rmse'
# param['base_score']  = 0.0
# num_round            = 25
# bst = xgb.train(param, dataset, num_round, evallist)
# bst.save_model('recsys2017.model')


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

# bucket_size = len(target_items) / N_WORKERS
# start = 0
# jobs = []
# for i in range(0, N_WORKERS):
#     stop = int(min(len(target_items), start + bucket_size))
#     filename = "solution_" + str(i) + ".csv"
#     process = multiprocessing.Process(target = classify_worker, args=(target_items[start:stop], target_users, items, users, filename, bst))
#     jobs.append(process)
#     start = stop
#
# for j in jobs:
#     j.start()
#
# for j in jobs:
#     j.join()

