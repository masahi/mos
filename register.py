import os
import logging
from conf import *

def register_all(target, atlas):
    for d in atlas:
        # print "Registration " + d
        if d == target: continue
        os.system("%s -f %s/vol.nii -m %s/vol.nii -o %s/demons_vol.nii --registrationFilterType Diffeomorphic -s 1 -n 5 -i 300,50,30,20,15 -O %s/demons_deformation.nii" % (BRAINSDemonWarp, target, d, d, d))
        os.system("%s --inputVolume %s/re_label.nii --deformationVolume %s/demons_deformation.nii --outputVolume %s/demons_label.nii --referenceVolume %s/re_label.nii --interpolationMode NearestNeighbor --pixelType uchar" % (BRAINSResample, d, d, d, target))
