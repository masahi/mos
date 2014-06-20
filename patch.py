import leveldb
import os
import numpy as np
import caffe_pb2
import random
import nibabel as nib
from skimage.segmentation import slic
from collections import defaultdict

patch_size = 29
rad = patch_size / 2

#db_xy = leveldb.LevelDB('test_db_xy')
#db_yz = leveldb.LevelDB('test_db_yz')
db_xz = leveldb.LevelDB('test_db_xz')

label_map = defaultdict(int)
label_map[9]  = 1
label_map[8] = 2
label_map[13] = 3
label_map[14] = 4
label_map[15] = 5
label_map[16] = 6
label_map[18] = 7
n_labels = 8
sample_freq = [0.002, 0.5, 0.5, 0.5, 0.5, 0.5,0.5,0.5]
sample_count = np.zeros(n_labels)
train_count = 0

for d in os.listdir("."):
    print d
    if not d.startswith("t00") : continue
    if not "190" in d: continue
    
    volume = nib.load(d + "/vol.nii").get_data().astype(np.float)
    label = nib.load(d + "/label.nii").get_data().astype(np.float)
    
    volume = np.ascontiguousarray(volume.swapaxes(1,2).swapaxes(0,1))
    label = np.ascontiguousarray(label.swapaxes(1,2).swapaxes(0,1))
    
    d, h, w = volume.shape
    size = d * h * w
    volume = np.pad(volume, (rad), "symmetric")
    
    for z in range(rad, d+rad):
        for y in range(rad, h+rad):
            for x in range(rad, w+rad):
                l = label_map[label[z-rad, y-rad, x-rad]]
                
                if random.random() > sample_freq[l]: continue
        
                # xy_patch = np.ascontiguousarray(volume[z, y-rad:y+rad+1, x-rad:x+rad+1].flatten() / 2048)
                # datum_xy = caffe_pb2.Datum()
                # datum_xy.channels = 1
                # datum_xy.height = datum_xy.width = patch_size
                # datum_xy.label = int(l)
                # datum_xy.float_data.extend(xy_patch.flat)
                # db_xy.Put(str(train_count), datum_xy.SerializeToString())

                # yz_patch = np.ascontiguousarray(volume[z-rad:z+rad+1, y-rad:y+rad+1, x].flatten() / 2048)
                # datum_yz = caffe_pb2.Datum()
                # datum_yz.channels = 1
                # datum_yz.height = datum_yz.width = patch_size
                # datum_yz.label = int(l)
                # datum_yz.float_data.extend(yz_patch.flat)
                # db_yz.Put(str(train_count), datum_yz.SerializeToString())

                xz_patch = np.ascontiguousarray(volume[z-rad:z+rad+1, y, x-rad:x+rad+1].flatten() / 2048)
                datum_xz = caffe_pb2.Datum()
                datum_xz.channels = 1
                datum_xz.height = datum_xz.width = patch_size
                datum_xz.label = int(l)
                datum_xz.float_data.extend(xz_patch.flat)
                db_xz.Put(str(train_count), datum_xz.SerializeToString())

                train_count += 1
                sample_count[l] += 1
                
print sample_count
print train_count
