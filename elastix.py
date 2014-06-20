import os

data_dir = "/home/masa/project/nii"
meta2nii = "/home/masa/project/itkutil/build2/meta2nii"

for d in os.listdir(data_dir):
    if os.path.exists("%s/%s/registered_label.nii" % (data_dir, d)): continue    
    print d
    if not d.startswith("t00") : continue
    if "190" in d: continue

    os.system("mkdir %s/%s/res" % (data_dir,d))
    os.system("elastix -f %s/t0000190_6/vol.nii -m %s/%s/vol.nii -out %s/%s/res -p parameters_BSpline.txt" % (data_dir, data_dir, d, data_dir, d))
    f = open("%s/%s/res/TransformParameters.0.txt" % (data_dir,d)).read()
    s = f.replace("FinalBSplineInterpolationOrder 3","FinalBSplineInterpolationOrder 0")
    tmp = open("%s/%s/res/param.txt" % (data_dir,d),"w")
    tmp.write(s)
    tmp.close()
    os.system("transformix -in %s/%s/re_label.nii -out %s/%s/res/ -tp %s/%s/res/param.txt" % (data_dir,d,data_dir, d,data_dir, d))
    os.system("%s %s/%s/res/result.mhd %s/%s/registered_label.nii" % (meta2nii, data_dir, d, data_dir, d))
    


