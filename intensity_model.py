from conf import *
import numpy as np
from joblib import Parallel, delayed,dump
import time
from nibabel import load
import helper
from sklearn.ensemble import RandomForestClassifier

def job(dir):    

    patches = []
    labels = []
    sample_count = np.zeros(n_labels) 
    
    rad = 2
    vol = load(dir + "/vol.nii")
    label = load(dir + "/re_label.nii")

    vol_data = np.ascontiguousarray(vol.get_data())
    label_data = np.ascontiguousarray(label.get_data())

    return helper.get_feature_subsample(vol_data.astype(np.float32), label_data, rad, n_labels, sample_freq)

def learn_forest(a, atlas):
    r = Parallel(n_jobs=len(atlas)-1)(delayed(job)(d) for d in atlas if a != d)
    features, labels, sample_count = zip(*r)
    features = np.vstack(features)
    labels = np.concatenate(labels)
    sample_count = reduce(lambda x,y: x+y, sample_count)
    print sample_count
    n_trees = 20
    forest = RandomForestClassifier(n_trees, n_jobs=n_trees)
    
    print "Training forest...",
    t = time.time()
    weight = 1.0/sample_count[labels]
    forest.fit(features, labels, sample_weight=weight)
    print time.time() - t
    # print "done."
    dump(forest, "forest.joblib.dump")
