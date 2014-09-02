import numpy as np
import nibabel as nib
from unary_feature import *
import time
from joblib import Parallel, delayed,load,dump
from nibabel import Nifti1Image
import os
from random import random
from conf import *

def get_beta(vol):
    d,h,w = vol.shape
    size = w * h * d
    grad_x, grad_y, grad_z = np.gradient(vol)
    beta = 1.0 / (2 * ((np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_z ** 2)) / (size * 3)))
    return beta

def fusion_move(unary, pair_costs, pair_index):
    fused = np.argmin(unary, axis=1)

    for ii in range(1):
        for i in range(n_labels):
            print i
            proposal = np.ones(fused.shape[0]) * i
            t = time.time()
            fused,energy,n_sup = helper.fusion_move(np.array(fused).astype(np.int32), proposal.astype(np.int32), np.array(unary).astype(np.float32), pair_costs, pair_index)
            print energy ,n_sup, pair_index.shape[0], time.time()-t

    return fused
            

def segment(a, atlas):
    vol = nib.load(a +  "/vol.nii")
    vol_data = vol.get_data()
    vol_data = np.ascontiguousarray(vol_data.swapaxes(0,2))
    d,h,w = vol_data.shape
    n_var = w * h * d
    epsilon = 1e-7
    unary_coeff = 1
    pair_coeff = 1.7
    
    t = time.time()
    
    beta = get_beta(vol_data)
    n_edge = (w - 1) * h * d + (h-1)*w*d + (d-1)*w*h
    pair_index = np.empty((n_edge, 2),dtype=np.uint32)
    pair_costs = np.empty((n_edge, n_labels, n_labels), np.float32)
    helper.get_edge_cost(vol_data, w,h,d, beta, pair_coeff, pair_index, pair_costs, 32)
    
    pa = get_atlas_term(vol_data, n_labels)
    patch_size = 5
    rad = patch_size / 2
    padded = np.pad(vol_data, rad, mode="constant", constant_values=(-2048))
    features = np.empty((n_var, patch_size**3), dtype=np.float32)
    
    n_threads = 32
    t = time.time()
    helper.get_feature(padded.astype(np.float32), features, rad, n_threads)
    print time.time() - t
    
    forest = load("forest.joblib.dump")
    forest.n_jobs = 5
    print forest.n_jobs
    t = time.time()
    prob = forest.predict_proba(features)
    print time.time() - t
    t = time.time()
    
    inten_w = 0.3
    atlas_weight = 1.0
    intensity_term = -inten_w * np.log(prob + epsilon)
    atlas_term = -atlas_weight * np.log(pa+epsilon)
    unary = atlas_term + intensity_term
    
    fused = fusion_move(unary, pair_costs, pair_index)
    
    seg_image = Nifti1Image(np.array(fused).reshape(w,h,d,order='F'), vol.get_affine())
    seg_image.to_filename(a + "/seg_label.nii")
    
