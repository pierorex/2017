import numpy as np
import multiprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from model import *
from parser import *
from recommendation_worker import *

N_WORKERS         = 4
DATA_FOLDER       = '../../xing/data/recsys'
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
data   = np.array([interactions[key].features() for key in interactions.keys()])
labels = np.array([interactions[key].label() for key in interactions.keys()])

'''
3) Train XGBoost regression model with maximum tree depth of 2 and 25 trees
'''
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
unique, counts = np.unique(labels, return_counts=True)
print('unique values in labels: {}'.format(dict(zip(unique, counts))))

# model = GaussianNB()
model = RandomForestClassifier(max_depth=14, random_state=42)
model.fit(X_train, y_train)

# test accuracy on test dataset
y_pred = model.predict(X_test)
unique, counts = np.unique(y_test, return_counts=True)

print('tn, fp, fn, tp = {}'.format(confusion_matrix(y_test, y_pred).ravel()))

incorrect = sum(p != l for p, l in zip(y_pred, y_test))
correct = len(y_test) - incorrect
accuracy = correct / len(y_test)
print('incorrect: {} | correct: {} | accuracy: {}'.format(incorrect, correct, accuracy))
# acc:
exit()
bst = model  # to follow the same pattern as before


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

