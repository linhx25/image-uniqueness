import numpy as np
import pandas as pd
import h5py
from PIL import Image
import glob

img_folder = "./output/data/ps/*"
output_path = "./output/data/"

# 1. Store all images to append (e.g. PS images) in img_folder
img_ids = []
for path in list(glob.glob(img_folder)):
    t = path.split("/")[-1]
    property_id = t.split("-")[0]
    image_id = ''.join(t.split("-")[1:])
    img_ids.append((property_id, image_id, path))

# 2. create new hdf5 file
ds = h5py.File(output_path + "/ps_photos.hdf5", 'a')
for property_id, image_id, path in img_ids:
    img = Image.open(path)
    img = img.resize((224,224))
    img = np.array(img)
    if property_id not in ds.keys():
        g = ds.create_group(property_id)
    else:
        g = ds[property_id]
    g[image_id] = img
ds.close()

# 3. update inference index file (We need not to infer all image)
img_ids = []
for path in list(glob.glob(img_folder)):
    t = path.split("/")[-1]
    property_id = t.split("-")[0]
    image_id = ''.join(t.split("-")[1:])
    img_ids.append((property_id, image_id))
img_ids = pd.MultiIndex.from_tuples(img_ids)
data_append = pd.DataFrame({"in/out": np.zeros(len(img_ids))}, index=img_ids).astype(int)
scene = pd.read_pickle("./output/data/debug_NY_idx.pkl")
scene = pd.concat([scene, data_append],axis=0).iloc[-256*5:, :]
scene.to_pickle(output_path + "/infer_ps_NY_idx.pkl")
