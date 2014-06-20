import numpy as np
from nibabel import load, Nifti1Image
import os
from collections import defaultdict
from random import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import opengm
import cPickle
from cPickle import dump


def add_pairwise_constraint(pairwise, LARGE_CONSTANT = 1e7):
    n_labels = pairwise.shape[0]
    pairwise[np.arange(n_labels), np.arange(n_labels)] = 0
    pairwise[3,4] = pairwise[3,4] = LARGE_CONSTANT

    return pairwise

def get_beta(vol):
    w,h,d = vol.shape
    size = w * h * d
    grad_x, grad_y, grad_z = np.gradient(vol)
    beta = 1.0 / (2 * ((np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_z ** 2)) / (size * 3)))
    return beta

def get_feature(vol):
    w,h,d = vol.shape
    size = w * h * d
    features = np.empty((size, 4))
    index = 0
    for z in range(d):
        for y in range(h):
            for x in range(w):
                feature = np.array([value, float(x)/w, float(y)/h, float(z)/d])
                features[index] = feature

    return features

n_labels = 8
label_map = defaultdict(int)
label_map[9]  = 1 #Gallbladder
label_map[8] = 2 #liver
label_map[13] = 3 #spleen
label_map[14] = 4 #right kidney
label_map[15] = 5 #left kidney
label_map[16] = 6 #IVC
label_map[18] = 7 #pancreas
                
d = "t0000190_6"
vol = load(d + "/vol.nii")
label = load(d + "/label.nii")

vol_data = vol.get_data()
label_data = label.get_data()

w,h,d = vol_data.shape

with open("forest.dump") as f:
    forest = cPickle.load(f)

n_var = w * h * d
n_labels = 8
gm = opengm.gm(np.ones(n_var, dtype=opengm.label_type)*n_labels)

epsilon = 1e-7

unary_coeff = 1
pair_coeff = 1

beta = get_beta(vol_data)
print beta

for z in range(d):
    print z
    for y in range(h):
        for x in range(w):
            
            l = label_map[label_data[x,y,z]]
            value = vol_data[x,y,z]

            feature = np.array([value, float(x)/w, float(y)/h, float(z)/d])
            feature = (feature - np.array([6.81791716, 0.47343367, 0.49059294, 0.54293764])) / np.array([  2.50522012e+02,   2.19092347e-01,   1.71592833e-01, 1.35432788e-01])

            index = x + y * w + z * w * h
            prob = forest.predict_proba(feature)
#            unary[index] = unary_coeff * -np.log(prob + epsilon)
                  
            fid = gm.addFunction(-np.log(prob+epsilon))
            
            gm.addFactor(fid, [index])

            if x != w-1:
                index_n = index + 1
                value_n = vol_data[x+1, y, z]
                cost = pair_coeff * np.exp(-(value-value_n)**2 * beta)
                pairwise = np.ones((n_labels, n_labels)) * cost
                pairwise = add_pairwise_constraint(pairwise)
                fid = gm.addFunction(pairwise)
                gm.addFactor(fid, [index, index_n])

            if y != h-1:
                index_n = index + w
                value_n = vol_data[x, y+1, z]
                cost = pair_coeff * np.exp(-(value-value_n)**2 * beta)                
                pairwise = np.ones((n_labels, n_labels)) * cost
                pairwise = add_pairwise_constraint(pairwise)                
                fid = gm.addFunction(pairwise)
                gm.addFactor(fid, [index, index_n])

            if z != d-1:
                index_n = index + w*h
                value_n = vol_data[x, y, z+1]
                cost = pair_coeff * np.exp(-(value-value_n)**2 * beta)                
                pairwise = np.ones((n_labels, n_labels)) * cost
                pairwise = add_pairwise_constraint(pairwise)                
                fid = gm.addFunction(pairwise)
                gm.addFactor(fid, [index, index_n])
                
            # labels.append(l)
            # features.append([value, float(x)/w, float(y)/h, float(z)/d])

opengm.saveGm(gm, "gm.h5")
# fids = gm.addFunctions(unary)
# gm.addFactors(fids, np.arange(0,n_var), dtype=np.uint64)
# seg_image = Nifti1Image(seg, label.get_affine(), header = label.get_header())
# seg_image.to_filename("seg.nii")


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

