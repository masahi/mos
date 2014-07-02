import numpy as np
from nibabel import Nifti1Image
import nibabel as nib
import os
from collections import defaultdict
from random import random
from sklearn.ensemble import RandomForestClassifier 
from sklearn import preprocessing
import time
from joblib import Parallel, delayed,load,dump
import helper
from unary_feature import *

epsilon = 1e-5

def get_beta(vol):
    d,h,w = vol.shape
    size = w * h * d
    grad_x, grad_y, grad_z = np.gradient(vol)
    beta = 1.0 / (2 * ((np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_z ** 2)) / (size * 3)))
    return beta

def fusion_move(pa, intensity_term, pair_costs, pair_index, atlas,w,h,d):
    # fusion_mover= opengm.inference.adder.minimizer.FusionMover(gm)
    n_labels = pa.shape[1]
    atlas_weight = 1
    prox_weight = 0.3

    atlas_term = -atlas_weight * -np.log(pa+epsilon)
    prox = get_prox_term(pa, np.zeros((d,h,w)), [1,1,1], n_labels, True)
    prox_term = -prox_weight * -np.log(prox+epsilon)
    a_i = atlas_term  + intensity_term
    unary =  a_i + prox_term
    fused = np.argmin(unary, axis=1)
    for i in range(n_labels):
        print i        
        label = np.ones(fused.shape[0]) * i

        fused,energy,n_sup = helper.fusion_move(np.array(fused).astype(np.int32), label.astype(np.int32), np.array(unary).astype(np.float32), pair_costs, pair_index)
        print dir,energy ,n_sup, pair_index.shape[0], time.time()-t
        prox = get_prox_term(pa, np.array(fused).reshape(d,h,w), [1,1,1], n_labels)
        prox_term = -prox_weight * np.log(prox + epsilon)
        unary = a_i + prox_term
        
#     for i,dir in enumerate(atlas):
#         t = time.time()
# #        label = nib.load(dir + "/re_label.nii")
#         label = nib.load(dir + "/registered_label.nii")        
#         fused,energy,n_sup = helper.fusion_move(np.array(fused).astype(np.int32), label.get_data().flatten(order="F").astype(np.int32), np.array(unary).astype(np.float32), pair_costs, pair_index)
#         print dir,energy ,n_sup, pair_index.shape[0], time.time()-t               
#         seg_image = Nifti1Image(np.array(fused).reshape(w,h,d,order='F'), label.get_affine(), header = label.get_header())
#         seg_image.to_filename("%s_seg.nii" % str(i))

    return fused

data_dir = "/home/masa/project/nii"
n_labels = 8
label_map = defaultdict(int)
label_map[9]  = 1 #Gallbladder
label_map[8] = 2 #liver
label_map[13] = 3 #spleen
label_map[14] = 4 #right kidney
label_map[15] = 5 #left kidney
label_map[16] = 6 #IVC
label_map[18] = 7 #pancreas
                
dir = "t0000190_6"
vol = nib.load(data_dir+ "/" + dir + "/vol.nii")
label = nib.load(data_dir+ "/" + dir + "/re_label.nii")

vol_data = vol.get_data()
label_data = label.get_data()

vol_data = np.ascontiguousarray(vol_data.swapaxes(0,2))
d,h,w = vol_data.shape
n_var = w * h * d
epsilon = 1e-7
unary_coeff = 1
pair_coeff = 2

t = time.time()
scaler = load("scaler.joblib.dump")

# features = helper.get_feature(vol_data.astype(np.float32))
# features = scaler.transform(features)    
# print time.time() - t
# print "get features"

# forest = load("forest.joblib.dump")

# print "loaded forest"
import cPickle
with open("atlas_list") as f:
    atlas = cPickle.load(f)

beta = get_beta(vol_data)
n_edge = (w - 1) * h * d + (h-1)*w*d + (d-1)*w*h
pair_index = np.empty((n_edge, 2),dtype=np.uint32)
pair_costs = np.empty((n_edge, n_labels, n_labels), np.float32)
helper.get_edge_cost(vol_data, w,h,d, beta, pair_coeff, pair_index, pair_costs, 32)

pa = get_atlas_term(vol_data, n_labels)
patch_size = 5
rad = patch_size / 2
padded = np.pad(vol_data, rad, mode="constant", constant_values=(-2048))

n_threads = 32
t = time.time()
features = helper.get_feature(vol_data.astype(np.float32))
print time.time() - t

forest = load("forest3.joblib.dump")
prob = forest.predict_proba(features)
dump(prob, "prob2.dump")
#prob = load("prob.dump")
# u = np.argmax(prob, axis=1)
# # current = np.argmax(pa,axis=1).reshape(d,h,w).astype(np.int32)
# # t = time.time()
# inten_w = 0.5
# intensity_term = -inten_w * np.log(prob + epsilon) 
# fused = fusion_move(pa,intensity_term, pair_costs, pair_index, atlas,w,h,d)
# seg_image = Nifti1Image(u.reshape(w,h,d,order='F'), label.get_affine(), header = label.get_header())
# seg_image.to_filename("unary_label.nii")
# print time.time() - t

# t = time.time()
# unary = -(atlas_weight * np.log(pa+epsilon) + prox_weight * np.log(prox+epsilon))
# print time.time() - t
#unary = -(np.log(((pa+epsilon)**atlas_weight) * (prox+epsilon)**prox_weight))


#unary = get_unary(vol_data, 1, n_labels)
# t = time.time()
# unary = unary_coeff * -np.log(forest.predict_proba(features) + epsilon)
# unary = unary.astype(np.float32)
# print time.time() - t

# print "got unary"
# #dump(unary,"unary.joblib.dump")
# #unary = load("unary.joblib.dump")

# current = np.argmin(unary, axis=1)
# seg_image = Nifti1Image(current.reshape(w,h,d,order="F"), label.get_affine(), header = label.get_header())
# seg_image.to_filename("unary.nii")


# print len(atlas)
         
# t = time.time()
# gm = opengm.gm(np.ones(n_var, dtype=opengm.label_type)*n_labels)
# fids = gm.addFunctions(pair_costs)
# gm.addFactors(fids, pair_index)
# print time.time() - t

# fids = gm.addFunctions(unary)
# gm.addFactors(fids, np.arange(0,n_var), dtype=np.uint64)
# opengm.saveGm(gm, "gm.h5")# 


# sol = fusion_move_fast(current)
# seg_image = Nifti1Image(sol.reshape(w,h,d,order='F'), label.get_affine(), header = label.get_header())
# seg_image.to_filename("seg_fast.nii")
# def fuse_job(current, fusion_mover, ds):
#     fused = current
#     for i,d in enumerate(ds):
#         label = load(d + "/label.nii").get_data().flatten(order="F")
#         t = time.time()
#         fused, energy_after, _, _ = fusion_mover.fuse(fused, label)
#         print time.time()-t, d, energy_after
        
#     return fused


# def fusion_move_fast(current):
#     n_thread = 16
#     atlas = np.random.choice(atlas, 64, replace=False)
#     fusion_mover= opengm.inference.adder.minimizer.FusionMover(gm)      
#     r = Parallel(n_jobs=n_thread)(delayed(job)(current,fusion_mover, atlas[64/n_thread*i:64/n_thread*(i+1)]) for i in range(n_thread))
#     n = len(r)
    
#     while n > 1:
#         r = Parallel(n_jobs=n/2)(delayed(job2)(fusion_mover, r[2*i], r[2*i+1]) for i in range(n/2))        
#         n /= 2

#     assert(len(r) == 1)
#     return r[0]
