from nibabel import Nifti1Image
import nibabel as nib
import numpy as np
from conf import *

def build_pa(target, atlas):
    count = np.zeros(n_labels)
    vol = nib.load(target + "/vol.nii")
    w,h,d = vol.shape
    pa = np.zeros((n_labels, w, h, d))
    
    for d in atlas:
        if target == d: continue
        lab = nib.load(d + "/demons_label.nii").get_data()        
        for l in range(n_labels):
            mask = lab == l
            if np.sum(mask) > 0:
                pa[l] += mask
                count[l] += 1
    
    for l in range(n_labels):
        pa[l] /= count[l]
        pa_image = Nifti1Image(pa[l], vol.get_affine())
        pa_image.to_filename("%s_pa.nii" % str(l)) 
        
