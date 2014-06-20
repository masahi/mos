from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from nibabel import load
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

data_dir = "/home/masa/project/nii"
p = load("seg.nii").get_data().flatten()
t = load(data_dir + "/t0000190_6/re_label.nii").get_data().flatten()
np.mean(p.flatten() == t.flatten())
names = ["BackGround",
    "Gallbladder",
 "Liver",
 "Spleen",
 "Right Kidney",
 "Left Kidney",
 "IVC",
"Pancreas"]

cm = confusion_matrix(p.flatten(), t.flatten())
prob = cm/ np.sum(cm, axis=1)[:,np.newaxis].astype(np.float)
print np.diag(prob)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(prob)
fig.colorbar(cax)
ax.set_yticklabels([''] + names)
#ax.set_yticklabels([''] + labels)
pl.xlabel("Prediction")
pl.ylabel('Ground Truth')
pl.show()




