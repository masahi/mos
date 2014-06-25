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
    return [img]
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
    for i in range(n_labels):
        features.append([])
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
                features[l].append(feat)

#                features.append([value, float(x)/w, float(y)/h, float(z)/d])

                sample_count[l] += 1

    return features, sample_count

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
features, sample_count = zip(*r)

import cPickle
for l in range(n_labels):
    feat = []
    for f in features:
        for ff in f[l]:
            feat.append(ff)
    if l == 0:
        gmm = GMM(5)
    else:
        gmm = GMM()
    gmm.fit(feat)
    with open("%s.gmm" % l, "w") as f:
        cPickle.dump(gmm, f)
# all_features = np.vstack(features)
# all_labels = np.concatenate(labels)
# features = np.array(all_features)
# sample_count = reduce(lambda x,y: x+y, sample_count)
# print sample_count
# print time.time() - t

# scaler = preprocessing.StandardScaler().fit(all_features)
# features = scaler.transform(all_features)
# forest = RandomForestClassifier()

# print "Training forest...",
# t = time.time()
# weight = 1.0/sample_count[labels]
# forest.fit(features, labels, sample_weight=weight)
# print time.time() - t
# print "done."

# dump(forest, "forest.joblib.dump")
# dump(scaler, "scaler.joblib.dump")
