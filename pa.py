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

w,h,d = label.shape
pa = np.zeros((n_labels, w, h, d))
import cPickle
with open("atlas_list") as f:
    atlas = cPickle.load(f)
    
count = np.zeros(n_labels)
from joblib import Parallel, delayed,dump
from numpy import corrcoef

def job(a):
    print a
    v2 = nib.load(a + "/vol.nii").get_data()
    cc = corrcoef(vol_data.flatten(), v2.flatten())
    return cc[0,1]
    
ccs = Parallel(n_jobs=len(atlas))(delayed(job)(a) for a in atlas)

s = np.argsort(ccs)

for i in range(10):
#    if s[i] > len(atlas)/2:continue
    lab = nib.load(atlas[s[-i-1]] + "/registered_label.nii").get_data()
    for l in range(n_labels):
        mask = lab == l
        if np.sum(mask) > 0:
            pa[l] += mask
            count[l] += 1

for l in range(n_labels):
    pa[l] /= count[l]
    pa_image = Nifti1Image(pa[l], label.get_affine(), header = label.get_header())
    pa_image.to_filename("%s_pa.nii" % str(l)) 
    
            
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
