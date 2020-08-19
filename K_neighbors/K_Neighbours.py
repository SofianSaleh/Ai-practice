import numpy as np
import warnings
import collections
import pandas as pd
import random

# k = features r = labels
dataset = {'k': [[1, 2], [2, 3], [3, 1]], "r": [[6, 5], [7, 7], [8, 6]]}
new_feature = [5, 7]

# equevilatn to
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# [[plt.scatter(ii[0], ii[1], s=100, color=i)
#   for ii in dataset[i]] for i in dataset]

# plt.show()


def k_nearest_neighbor(data, predict, k=3):
    if k >= len(data):
        warnings.warn('k is wrong man!!')
    distances = []
    for group in data:
        for features in data[group]:
            # calculate the eucladian distance
            euclidean_distance = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = collections.Counter(votes).most_common(1)[0][0]
    confidence = collections.Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

# test purpos
# acc = []
# for i in range(10):


df = pd.read_csv('./K_neighbors/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
# dropping the unneccecry field id
df.drop(['id'], 1, inplace=True)
#  redefinig the data so it all can be float
full_data = df.astype('float').values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
# start from the begining untill the last 20%
#  if there is no "-" then it will take 20%
train_data = full_data[:-int(test_size * len(full_data))]
# capture the last 20% only
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    # append all except the class
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    # append all except the class
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
# in this for loop we are going to find the accuracy simply from above we know in which group the data belongs to
# so we test our k_nearest_neighbour function by making it classifiy the data and then comparing the data to its key
# {2:[[1,2,1,1,2]]}
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbor(test_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1


print('accuracy: ', correct/total)
# test purpos
# # acc.append(correct/total)
# print(sum(acc)/len(acc))

