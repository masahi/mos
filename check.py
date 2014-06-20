from nibabel import load, Nifti1Image
import numpy as np
import os
data_dir = "/home/masa/project/nii"

count = 0
for d in os.listdir(data_dir):


    if os.path.exists("%s/%s/registered_label.nii" % (data_dir, d)): count +=1


print count
