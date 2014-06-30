import nibabel as nib
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import LogisticRegression
import cPickle
import time
from skimage.util import pad

vol_file = "/home/masa/t0003743_4/vol.nii.gz"
label_file = "/home/masa/t0003743_4/label.nii.gz"

vol = nib.load(vol_file).get_data()
label_map = nib.load(label_file).get_data()

w,h,d = vol.shape

def get_patch3d(volume, size, subsample=1):
    patches = []
    labels = []
    rad = size/2
    for z in range(rad,d-rad,subsample):
        for y in range(rad,h-rad,subsample):
            for x in range(rad,w-rad,subsample):
                p = volume[x-rad:x+rad+1, y-rad:y+rad+1, z-rad:z+rad+1].flatten()
                l = label_map[x,y,z]
                patches.append(p)
                labels.append(l)

    return np.vstack(patches), np.vstack(labels)

size = 5
subsample = 5
t = time.time()
patches,labels = get_patch3d(vol, size, subsample)
print time.time() - t

n_jobs = 32
estimator = DictionaryLearning(200, n_jobs = n_jobs)
estimator.fit(patches)
# clf = LogisticRegression()
# clf.fit(X, labels)

# padded = pad(vol, size/2, mode="symmetric")
# patches = get_patch3d(padded, size, 1)

# X_test = estimator.transform(patches)
# y = label_map.flatten()
# prediction = clf.predict(X)

# print np.mean(prediction == y)




