import leveldb
from caffe_pb2 import Datum
from caffe import Net
import numpy as np
import nibabel as nib
import time
from collections import defaultdict
volume = nib.load("vol.nii.gz").get_data()
label = nib.load("label.nii.gz").get_data()

model_def_file = "net.proto"
trained_model = "lenet_iter_100000"
caffenet = Net(model_def_file, trained_model)
#caffenet.set_mode_gpu()

acc = 0
c = 0
per_class = np.array([0,0,0,0,0], dtype=np.float)
class_count = np.array([0,0,0,0,0], dtype=np.float)


patch_size = 9
rad = patch_size / 2

volume = volume.swapaxes(1,2).swapaxes(0,1)
d,h,w = volume.shape
volume = np.pad(volume, (rad), "symmetric")
volume = volume.astype(np.float32)/2048

seg = np.zeros((d,h,w), dtype=np.uint8)
label_map = defaultdict(int)
label_map[8] = 1
label_map[13] = 2
label_map[14] = 3
label_map[15] = 3
label_map[18] = 4

t = time.time()
true_label = []
pred_label = []
output_blob = [np.zeros((1, 5,1,1), dtype=np.float32)]

for z in range(rad, d+rad):
    print z
    
    index = 0    
    for y in range(rad, h+rad):
        for x in range(rad, w+rad):
            patch = volume[z-rad:z+rad+1, y-rad:y+rad+1, x-rad:x+rad+1]

            input_blob = [np.ascontiguousarray(patch.reshape(1, patch_size, patch_size, patch_size))]
                
            caffenet.Forward(input_blob, output_blob)
            prediction = np.argmax(output_blob[0][0])

            seg[z-rad, y-rad, x-rad] = prediction
        
seg = seg.swapaxes(1,2).swapaxes(0,1)
seg_image = nib.Nifti1Image(seg, label.get_affine(),header =label.get_header())
seg_image.to_filename("seg2.nii.gz")
# print "forwarding... "                            
# np.save("all_patches.npy". all_patches)
# np.save("all_labels.npy". all_labels)
# output_blob = [np.empty((w*h*d, 5,1,1), dtype=np.float32)]
# input_blob = [np.ascontiguousarray(all_patches)]
# caffenet.set_mode_gpu()
# caffenet.Forward(input_blob, output_blob)
# np.save("output_blob.npy", output_blob[0])
# prediction = np.argmax(output_blob[0], axis=1).flatten()
# print np.mean(all_labels == prediction)
# print time.time() -t

# for key,item in db.RangeIter():
#     data = Datum()
#     data.ParseFromString(item)
#     output_blob = [np.empty((1, 5,1,1), dtype=np.float32)]
#     patch = np.asarray(data.float_data, dtype=np.float32) / 2048
#     input_blob = [np.ascontiguousarray(patch.reshape(1,patch_size, patch_size, patch_size))]

#     caffenet.Forward(input_blob, output_blob)
#     prediction = np.argmax(output_blob[0][0])
#     # if prediction is not 0:
#     #         print prediction, data.label, output_blob[0][0].T   

#     if data.label == prediction:
#         acc += 1
#         per_class[prediction] += 1
#     c+=1
#     class_count[data.label] += 1
