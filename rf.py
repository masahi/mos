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

sample_freq = [0.00001,0.02, 0.00025, 0.003, 0.003, 0.003,0.015,0.015]

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

    features = []
    labels = []
    sample_count = np.zeros(n_labels) 
    
    
    vol = load(dir + "/vol.nii")
    label = load(dir + "/label.nii")

    vol_data = vol.get_data()
    label_data = label.get_data()

    w,h,d = vol_data.shape
    bank = filter_bank(vol_data)
    for x in range(w):
        for y in range(h):
            for z in range(d):
                l = label_map[label_data[x,y,z]]
                if random() > sample_freq[l]: continue                                
#                l = label_data[x,y,z]
                feat = []
                for f in range(len(bank)):
                    value = bank[f][x,y,z]
                    feat.append(value)
                features.append(feat)
                label.append(l)

#                features.append([value, float(x)/w, float(y)/h, float(z)/d])

                sample_count[l] += 1

    return features, label, sample_count

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
r = Parallel(n_jobs=25)(delayed(job)(data_dir + "/" + d) for d in os.listdir(data_dir) if d.startswith("t00") and not "190" in d)
features, labels, sample_count = zip(*r)
features = np.vstack(features)
labels = np.concatenate(labels)
sample_count = reduce(lambda x,y: x+y, sample_count)

import cPickle
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
forest = RandomForestClassifier()

# print "Training forest...",
t = time.time()
weight = 1.0/sample_count[labels]
forest.fit(features, labels, sample_weight=weight)
print time.time() - t
print "done."
dump(forest, "forest.joblib.dump")
dump(scaler, "scaler.joblib.dump")
