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

prob = load("prob.dump")
pa = load("pa.dump")
prox = load("prox.dump")

inten_weight = 0.3
atlas_weight = 0.8

prox_weight = 0.2

t = time.time()
unary = -inten_weight * np.log(prob+epsilon) -atlas_weight*np.log(pa+epsilon) - prox_weight * np.log(prox+epsilon)
print time.time() - t

sol = np.argmin(unary, axis=1)
seg_image = Nifti1Image(sol.reshape(w,h,d,order="F"), label.get_affine(), header = label.get_header())
seg_image.to_filename("unary2_label.nii")
# for l in range(n_labels):
#     pa_image = Nifti1Image(, label.get_affine(), header = label.get_header())
#     pa_image.to_filename("%s_pa.nii" % str(l)) 


