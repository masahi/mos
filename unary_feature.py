import numpy as np
import os
import nibabel as nib
import cPickle
from scipy.ndimage.morphology import distance_transform_edt
import helper
from joblib import Parallel, delayed,dump
import time

data_dir = "/home/masa/project/nii"
epsilon = 1e-5

def get_unary(img, current_labeling, atlas_weight, prox_weight, n_labels):
#    data_likelyhood = get_data_term(img, n_labels)
    prob_atlas = get_atlas_term(img, n_labels)
    prox_term = get_prox_term(img, current_labeling, n_labels)    
    return atlas_weight * -np.log(prob_atlas+epsilon) + prox_weight * -np.log(prox_term+epsilon)
 # #   data_likelyhood = get_data_term(img, n_labels)
    # prob_atlas = get_atlas_term(img, n_labels)

    # return  -np.log(prob_atlas+epsilon)

def prox_job(pa):
    pa_thres = 0.5    
    mask = pa > pa_thres
    t = time.time()
    dt= distance_transform_edt(1-mask)
    m = np.mean(dt)
    m = 50
    temp_mask = (dt > m)
    dt[np.logical_not(temp_mask)] /= float(m)
    dt[temp_mask] = 1
        
    return dt.flatten() / np.max(dt)

def prox_job2(pa):
    pa_thres = 0.5    
    mask = pa > pa_thres
    t = time.time()
    dt,ind= distance_transform_edt(1-mask,  return_indices=True)
    inds = ind.T.reshape(-1, 3)
    m = np.mean(dt)
    m = 50
    temp_mask = (dt > m)
    dt[np.logical_not(temp_mask)] /= float(m)
    dt[temp_mask] = 1
    
    return dt.flatten() / np.max(dt)
    
def get_prox_term(pa,current_labeling, spacing, n_labels):
    d,h,w = current_labeling.shape
    n = w * h * d
    prox_term = np.ones((n, n_labels))
    # if not first:
    #     binary_masks = np.zeros((n_labels, d, h, w), dtype=np.int32)
    #     helper.get_binary_mask_fast(current_labeling, binary_masks)

    t = time.time()
    r = Parallel(n_jobs=n_labels-1)(delayed(prox_job)(pa[:,i].reshape(d,h,w)) for i in range(1, n_labels))
    prox_term = np.hstack(r)
    print time.time() - t
    return 1 - prox_term
        
    
def get_feature(img):
    return img.flatten()

def get_data_term(img, n_labels):
    d,h,w = img.shape
    n = w * h * d
    data_term = np.zeros((n, n_labels))
    feature = get_feature(img)
    for l in range(n_labels):
        with open("%s.gmm" % l) as f:
            gmm = cPickle.load(f)
        import time
        t = time.time()
        prob = gmm.score_samples(feature)
        print time.time() - t
        data_term[:, l] = prob[0]
    return data_term

def get_atlas_term(img,n_labels):
    d,h,w = img.shape
    n = w * h * d    
    atlas_term = np.zeros((n, n_labels))
    for l in range(n_labels):
        pa_image = nib.load("%s_pa.nii" % str(l)).get_data()
        pa_image = pa_image.swapaxes(0,2)
        atlas_term[:, l] = pa_image.flatten()
    return atlas_term
    
    # n = w * h * d
    # prob_atlas = np.zeros(n, n_labels)
    # atlas = [data_dir + "/" + dir for dir in  os.listdir(data_dir) if dir.startswith("t00")]
    # ssd_globals= []
    # selected_atlas = []
    # n_atlas_to_select = len(atlas) / 3
    # for a in atlas:
    #     vol = nib.load(a + "/vol.nii")
    #     ssd_globals.append(np.sum((img - vol)**2))
        
    # perm = np.argsort(ssd_globals)
    # for i,a in enumerate(atlas):
    #     if perm[i] > n_atlas_to_select: continue
    #     else selected_atlas.append(a)

    # w_l = np.zeros(
    # for a in selected_atlas:
    #     vol = nib.load(a + "/affine.nii")
    #     label = nib.load(a + "/affine_label.nii")

    #     for l in n_labels:
    #         mask = label == l
    #         ssd = np.sum((img[mask] - vol[mask])**2)
            
