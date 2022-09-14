import numpy as np
import h5py
from PIL import Image
import glob

img_folder = "./output/data/ps/*"
output_path = "./output/data/ps_photos.hdf5"

# 1. Store all images to append (e.g. PS images) in img_folder
img_ids = []
for path in list(glob.glob(img_folder)):
    t = path.split("/")[-1]
    property_id = t.split("-")[0]
    image_id = ''.join(t.split("-")[1:])
    img_ids.append((property_id, image_id, path))

# 2. create new hdf5 file
ds = h5py.File(output_path, 'a')
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
