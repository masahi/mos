import numpy as np

n_labels = 8
epsilon = 1e-5
data_dir = "/home/masa/project/nii"
BRAINSDemonWarp = "/home/masa/Slicer-4.3.1-linux-amd64/lib/Slicer-4.3/cli-modules/BRAINSDemonWarp"
BRAINSResample = "/home/masa/Slicer-4.3.1-linux-amd64/lib/Slicer-4.3/cli-modules/BRAINSResample"
sample_freq = np.array([0.001,1.0, 0.03, 0.5, 0.3, 0.3,1.0,1.0], dtype=np.float32)
