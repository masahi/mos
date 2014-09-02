from conf import *
from nibabel import load
import numpy as np

def compute_jaccard(a):
    p = load(a + "/seg_label.nii").get_data().flatten()
    t = load(a + "/re_label.nii").get_data().flatten()
    jac = []
    for i in range(n_labels):
        true = t.flatten() == i
        pred = p.flatten() == i
        num = np.sum(np.logical_and(true,pred))
        den = np.sum(np.logical_or(true, pred))
        jac.append(float(num)/float(den))
        
    return jac
    
    
