import numpy as np
import cPickle
from conf import *
from register import register_all
from build_pa import build_pa
from segment import segment
from jaccard import compute_jaccard
from intensity_model import learn_forest
import time

with open("atlas_list") as f:
    atlas = cPickle.load(f)

n_atlas = len(atlas)
jac = np.zeros((n_atlas, n_labels))

for i,a in enumerate(atlas):
    t = time.time()
    print "Segmenting " + a
    register_all(a, atlas)
    build_pa(a, atlas)
    learn_forest(a, atlas)
    segment(a, atlas)
    jac[i] = compute_jaccard(a)
    print jac[i]
    

np.save("jaccard_score.npy", jac)
mean_jac = np.mean(jac, axis=0)
print mean_jac
