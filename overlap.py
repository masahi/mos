from nibabel import load, Nifti1Image
import numpy as np

data_dir = "/home/masa/project/nii"
ref = "t0000190_6"
ref_label = load("%s/%s/re_label.nii" % (data_dir,ref)).get_data()
ref_bool = []
ref_count = []
for l in [1,2,3,4,5,6,7]:
    ref = (ref_label == l)
    ref_bool.append(ref)
    ref_count.append(np.sum(ref))

for d in os.listdir(data_dir):
    print d
    
    label = load("%s/%s/registered_label.nii" % (data_dir, d)).get_data()
    
    for l in [1,2,3,4,5,6,7]:
        lab = (label == l) 
        c = np.sum(ref_bool[l-1] & lab)
        print l, float(c)/ref_count[l-1]
