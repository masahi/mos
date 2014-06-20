import numpy as np
from nibabel import load, Nifti1Image
import os
from collections import defaultdict
from random import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import cPickle
from cPickle import dump

features = []
labels = []

n_labels = 8
label_map = defaultdict(int)
label_map[9]  = 1
label_map[8] = 2
label_map[13] = 3
label_map[14] = 4
label_map[15] = 5
label_map[16] = 6
label_map[18] = 7

d = "t0000190_6"
label = load(d + "/label.nii")

label_data = label.get_data()

w,h,d = label_data.shape

for z in range(d):
    print z
    for y in range(h):
        for x in range(w):
            l = label_map[label_data[x,y,z]]

            label_data[x,y,z] = l
            # labels.append(l)
            # features.append([value, float(x)/w, float(y)/h, float(z)/d])

seg_image = Nifti1Image(label_data, label.get_affine(), header = label.get_header())
seg_image.to_filename("relabeled.nii")


# features = np.array(features)
# labels = np.array(labels)
# features = preprocessing.scale(features)
# np.save("testing_features.npy", features)
# np.save("testing_labels.npy", labels)

# features = np.load("testing_features.npy")
# labels = np.load("testing_labels.npy")

# with open("forest.dump") as f:
#     forest = cPickle.load(f)

# prediction = forest.predict(features)
# print np.mean(prediction == labels)

