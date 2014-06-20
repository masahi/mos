import numpy as np
from nibabel import load, Nifti1Image
from skimage.segmentation import slic, find_boundaries
import time

vol = load("vol.nii")
label = load("label.nii")

vol_data = vol.get_data()
vol_data = np.ascontiguousarray(vol_data.swapaxes(1,2).swapaxes(0,1))
d, h, w = vol.shape
size = d * h * w
n_segments = 50
spacing = np.ascontiguousarray(np.diag(vol.get_affine()[:-1])[::-1])
t = time.time()
seg = slic(vol_data.astype(np.float64), n_segments, multichannel=False, spacing = spacing, convert2lab=False)
print time.time() - t

seg_image = Nifti1Image(seg.swapaxes(1,0).swapaxes(2,1), label.get_affine(), header = label.get_header())
mark_label = find_boundaries(seg)
mark_image = Nifti1Image(mark_label.swapaxes(1,0).swapaxes(2,1), label.get_affine(), header = label.get_header())
seg_image.to_filename("supervoxel_big_label.nii")
mark_image.to_filename("boundary_big_label.nii")

