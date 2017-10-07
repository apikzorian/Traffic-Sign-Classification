import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import csv
import random
import matplotlib.pyplot as plt
import pickle

import os
cwd = os.getcwd()


def color2gray(colorimg):
    gray = 0.2989 * colorimg[:, :, 0] + 0.5870 * colorimg[:, :, 1] + 0.1140 * colorimg[:, :, 2]
    return gray


def pre_process_image(pre_image):
    pre_image = (pre_image - 128.0) / 128.0
    return pre_image



pickle_file = 'T1P2.p'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  X_train = pickle_data['X_train']
  y_train = pickle_data['y_train']
  X_test = pickle_data['X_test']
  y_test = pickle_data['y_test']
  X_valid = pickle_data['X_valid']
  y_valid = pickle_data['y_valid']
  del pickle_data  # Free up memory

#
# signs = pd.read_csv('signnames.csv')
# signs.head()
#
# print(cwd)
# path = cwd + "/images/ahead-only.jpg"
#

# image = mpimg.imread(path)

# test_image = np.array([mpimg.imread('./ahead-only.jpg')])
#
# image = test_image[0]
# resized_image = cv2.resize(image, (32, 32))
# test_image = resized_image
# test_image = test_image.reshape(1,32,32,3)
# test_image = test_image.astype(dtype=np.float32)
# print(test_image.dtype)
#
#
# my_test_signs_labels = [35]


from scipy import misc
g_train = []

for i in range(5):
    j = i + 1
    path = cwd + '/images/german_sign_new_' + str(j) + '.jpg'
    img = misc.imread(path)
    resized_image = cv2.resize(img, (32, 32))
    g_train.append(resized_image)

fig, axs = plt.subplots(1,5)
axs = axs.ravel()
fig.suptitle('German Traffic Signs via Google Images', fontsize=20)
for i, img in enumerate(g_train):
    axs[i].axis('off')
    axs[i].imshow(img)
plt.show()


ger_features = np.array([pre_process_image(g_train[i]) for i in range(len(g_train))],
                          dtype = np.float32)

# Count frequency of each class in the train set


unique, counts = np.unique(y_train, return_counts=True)

ger_labels = [None] * 5

signname_map = np.genfromtxt('signnames.csv', delimiter=',', usecols=(1,), unpack=True, dtype=str, skip_header = 1)
predictions = 5
predict_signNames = []
for id in range(43):
    predict_signNames.append(signname_map[int(id)])
    signname = signname_map[int(id)]
    if(signname == "Speed limit (50km/h)"):
        ger_labels[0] = int(id)
    elif(signname == "Speed limit (80km/h)"):
        ger_labels[1] = int(id)
    elif(signname == "Roundabout mandatory"):
        ger_labels[2] = int(id)
    elif(signname == "Speed limit (30km/h)"):
        ger_labels[3] = int(id)
    elif(signname == "Speed limit (120km/h)"):
        ger_labels[4] = int(id)

g_labels = np.asarray(ger_labels)

a = {'g_train': ger_features, 'g_labels': g_labels}
with open('GermanSigns.p', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Pickle dumped for GermanSigns.p')