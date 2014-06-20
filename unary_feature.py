import numpy as np
import os
import nibabel as nib
data_dir = "/home/masa/project/nii"

def get_unary(img, atlas_weight,n_labels):
    data_likelyhood = get_data_term(img)
    prob_atlas = get_atlas_term(img, n_labels)
    return -np.log(data_likelyhood + atlas_weight * prob_atlas)
    
def get_data_term(img):
    pass

def get_atlas_term(img,n_labels):
    w,h,d = img.shape
    n = w * h * d
    prob_atlas = np.zeros(n, n_labels)
    atlas = [data_dir + "/" + dir for dir in  os.listdir(data_dir) if dir.startswith("t00")]
    ssd_globals= []
    selected_atlas = []
    n_atlas_to_select = len(atlas) / 3
    for a in atlas:
        vol = nib.load(a + "/vol.nii")
        ssd_globals.append(np.sum((img - vol)**2))
        
    perm = np.argsort(ssd_globals)
    for i,a in enumerate(atlas):
        if perm[i] > n_atlas_to_select: continue
        else selected_atlas.append(a)

    w_l = np.zeros(
    for a in selected_atlas:
        vol = nib.load(a + "/affine.nii")
        label = nib.load(a + "/affine_label.nii")

        for l in n_labels:
            mask = label == l
            ssd = np.sum((img[mask] - vol[mask])**2)
            

