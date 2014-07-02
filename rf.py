import numpy as np
from nibabel import load, Nifti1Image
import os
from collections import defaultdict
from random import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.mixture import GMM
#from cPickle import dump
import time
from joblib import Parallel, delayed,dump
from scipy.ndimage.filters import *

n_labels = 8
label_map = defaultdict(int)
label_map[9]  = 1
label_map[8] = 2
label_map[13] = 3
label_map[14] = 4
label_map[15] = 5
label_map[16] = 6
label_map[18] = 7

sample_freq = [0.001,0.1, 0.001, 0.015, 0.015, 0.015,0.015,0.015]

def filter_bank(img):
    bank = []
    sigma = [1, 2, 4, 8]
    bank.append(img)
    for s in sigma:
        bank.append(gaussian_filter(img, s))
    bank.append(laplace(img))
    return bank               
                    
    

def job(dir):    
    print dir

    patches = []
    labels = []
    sample_count = np.zeros(n_labels) 
    
    rad = 2
    vol = load(dir + "/vol.nii")
    label = load(dir + "/label.nii")

    vol_data = vol.get_data()
    label_data = label.get_data()

    w,h,d = vol_data.shape
    for x in range(rad, w-rad):
        for y in range(rad, h-rad):
            for z in range(rad, d-rad):
                l = label_map[label_data[x,y,z]]
                if random() > sample_freq[l]: continue                                
#                l = label_data[x,y,z]
                p = vol_data[x-rad:x+rad+1, y-rad:y+rad+1, z-rad:z+rad+1].flatten()
                patches.append(p)
                labels.append(l)

#                features.append([value, float(x)/w, float(y)/h, float(z)/d])

                sample_count[l] += 1

    return patches, labels, sample_count

def job2(dir):    
    print dir

    features = []
    labels = []
    sample_count = np.zeros(n_labels) 
    
    rad = 2
    vol = load(dir + "/vol.nii")
    label = load(dir + "/label.nii")

    vol_data = vol.get_data()
    label_data = label.get_data()

    vol_data = vol_data.swapaxes(0,2)
    label_data = label_data.swapaxes(0,2)    
    d,h,w = vol_data.shape
    for z in range(d):
        for y in range(h):
            for x in range(w):
                l = label_map[label_data[z,y,x]]
                if random() > sample_freq[l]: continue                                
#                l = label_data[x,y,z]
                features.append([vol_data[z,y,x], z, y, x])
                labels.append(l)

#                features.append([value, float(x)/w, float(y)/h, float(z)/d])

                sample_count[l] += 1

    return features, labels, sample_count

    # if not dir.startswith("t00"): return
    # print dir
    # label = load(dir + "/label.nii")

    # label_data = label.get_data()

    # w,h,d = label_data.shape

    # for x in range(w):
    #     for y in range(h):
    #         for z in range(d):
    #             l = label_map[label_data[x,y,z]]
    #             label_data[x,y,z] = l

    # seg_image = Nifti1Image(label_data, label.get_affine(), header = label.get_header())
    # seg_image.to_filename(dir + "/re_label.nii")
        
    return 

data_dir = "/home/masa/project/nii"
t = time.time()
import cPickle
with open("atlas_list") as f:
    atlas = cPickle.load(f)

r = Parallel(n_jobs=len(atlas))(delayed(job2)(a) for a in atlas)
features, labels, sample_count = zip(*r)
features = np.vstack(features)
labels = np.concatenate(labels)
sample_count = reduce(lambda x,y: x+y, sample_count)

dump(features, "features.dump")
dump(labels, "labels.dump")
# import cPickle
# # scaler = preprocessing.StandardScaler().fit(features)
# # features = scaler.transform(features)
print "Training"
n_trees = 20
forest = RandomForestClassifier(n_trees)

# print "Training forest...",
t = time.time()
weight = 1.0/sample_count[labels]
forest.fit(features, labels, sample_weight=weight)
print time.time() - t
# print "done."
dump(forest, "forest3.joblib.dump")
#dump(scaler, "scaler.joblib.dump")
