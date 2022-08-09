import torch
from torch.utils.data import DataLoader, Dataset
import glob
import h5py
import numpy as np
import pandas as pd
from PIL import ImageFilter
import random
import torchvision


class AirbnbDataLoader(DataLoader):
    ## TODO: add transform
    def __init__(
        self, root: str, batch_size=1, shuffle=True, sampler=None, drop_last=True
    ):
        if root.endswith(".h5"):
            self.files = [root]
        else:
            self.files = glob.glob(root + "/*.h5")
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.ds = [h5py.File(f, "r") for f in self.files]
        self.dt_len = [len(f["IDs"]) for f in self.ds]
        self.batch_slices = self._get_slices()
        self.idx_mapping = self._index2items()

    def _get_slices(self):
        batch_slices = [
            [
                slice(i, i + self.batch_size)
                for i in range(0, dt_len, self.batch_size)
                if i + self.batch_size <= dt_len
            ]
            for dt_len in self.dt_len
        ]
        if not self.drop_last:
            for i, slc in enumerate(self.batch_slices):
                last_idx = slc[-1].stop
                self.batch_slices[i].extend([slice(last_idx, self.dt_len[i])])
        return batch_slices

    def __len__(self):
        return sum(self.dt_len)

    def __iter__(self):
        all_slices = self.batch_slices.copy()
        if self.shuffle:
            for slices in all_slices:
                np.random.shuffle(slices)

        for i, slices in enumerate(all_slices):
            ds = self.ds[i]
            images, ids = ds["images"], ds["IDs"]
            for slc in slices:
                feature = images[slc].reshape(-1, 224, 224, 3).astype("uint8")
                label = ids[slc]
                feature = torch.tensor(feature, dtype=torch.float)
                yield (feature, label)

    def _index2items(self):
        idx_range = []
        stops = np.cumsum(self.dt_len)
        for i in range(len(stops)):
            if i == 0:
                idx_range.append(range(0, stops[i]))
            else:
                idx_range.append(range(stops[i - 1], stops[i]))

        idx = np.arange(stops[-1])
        res = pd.DataFrame(np.zeros((stops[-1], 3)), columns=["idx_file", "idx_image", "image_id"])
        for i in range(len(idx_range)):
            flag = np.isin(idx, idx_range[i])
            res.loc[flag, "idx_file"] = i
            res.loc[flag, "idx_image"] = idx[flag] - idx_range[i].start
            res.loc[flag, "image_id"] = self.ds[i]["IDs"][()]

        return res.astype(int)

    def __getitem__(self, idx: int):

        idx_file, idx_image = self.idx_mapping.loc[idx].values
        ds = self.ds[idx_file]
        images, ids = ds["images"], ds["IDs"]
        return (images[idx_image].reshape(224, 224, 3).astype("uint8"), ids[idx_image])


class AirbnbDataset(Dataset):
    def __init__(self, root: str, transform=None, image_ids=None):
        '''
        root: root path to the data
        transform: torch transformation
        image_ids: list, image ids to retrieve
        '''
        if root.endswith(".h5") or root.endswith(".hdf5"):
            self.files = [root]
        else:
            self.files = [
                t for t in glob.glob(root + "/*")
                if t.endswith(".h5") or t.endswith(".hdf5")
            ]

        self.ds = [h5py.File(f, "r") for f in self.files]
        self.dt_len = [len(f["IDs"]) for f in self.ds]
        self.idx_mapping = self._index2items()
        if image_ids is not None:
            self.idx_mapping = self.idx_mapping[self.idx_mapping["image_id"].isin(image_ids)]
            self.dt_len = list(self.idx_mapping["idx_file"].value_counts().sort_index())
        self.transform = transform

    def _index2items(self):
        idx_range = []
        stops = np.cumsum(self.dt_len)
        for i in range(len(stops)):
            if i == 0:
                idx_range.append(range(0, stops[i]))
            else:
                idx_range.append(range(stops[i - 1], stops[i]))

        idx = np.arange(stops[-1])
        res = pd.DataFrame(np.zeros((stops[-1], 3)), columns=["idx_file", "idx_image", "image_id"])
        for i in range(len(idx_range)):
            flag = np.isin(idx, idx_range[i])
            res.loc[flag, "idx_file"] = i
            res.loc[flag, "idx_image"] = idx[flag] - idx_range[i].start
            res.loc[flag, "image_id"] = self.ds[i]["IDs"][()]

        return res.astype(int)

    def __len__(self):
        return sum(self.dt_len)

    def __getitem__(self, idx: int):

        idx_file = self.idx_mapping.iat[idx, 0]
        idx_image = self.idx_mapping.iat[idx, 1]
        ds = self.ds[idx_file]
        images, ids = ds["images"], ds["IDs"]
        img = images[idx_image].reshape(224, 224, 3).astype("uint8")
        
        if self.transform:
            img = torchvision.transforms.ToPILImage()(img)
            img = self.transform(img)
        return {"image": img, "label": ids[idx_image],}


class AirbnbNYDataset(Dataset):
    def __init__(self, root: str, transform=None, image_ids=None):
        '''
        root: root path to the data
        transform: torch transformation
        image_ids: list, image ids to retrieve
        '''
        if root.endswith(".h5") or root.endswith(".hdf5"):
            self.files = [root]
        else:
            self.files = [
                t for t in glob.glob(root + "/*")
                if t.endswith(".h5") or t.endswith(".hdf5")
            ]

        self.ds = [h5py.File(f, "r") for f in self.files]
        self.idx_mapping = self._index2items()
        if image_ids is not None: 
            m = self.idx_mapping.set_index(["idx_image", "image_id"])
            self.idx_mapping = self.idx_mapping[m.index.isin(image_ids)]
        self.dt_len = list(self.idx_mapping["idx_file"].value_counts().sort_index())
        self.transform = transform
        
    def _index2items(self):
        
        res = []
        for i, f in enumerate(self.ds):
            for p in f.keys():
                id_img = f[p].keys()
                t = zip([i]*len(id_img), [p]*len(id_img), id_img)
                res.extend(list(t))
                
        return pd.DataFrame(res, columns=["idx_file", "idx_image", "image_id"])
    
    def __len__(self):
        return sum(self.dt_len)

    def __getitem__(self, idx: int):

        idx_file = self.idx_mapping.iat[idx, 0]
        idx_property = self.idx_mapping.iat[idx, 1]
        image_id = self.idx_mapping.iat[idx, 2]
        ds = self.ds[idx_file]
        img = ds[idx_property][image_id][()].astype("uint8")
        
        if self.transform:
            img = torchvision.transforms.ToPILImage()(img)
            img = self.transform(img)
        return {"image": img, "label": (idx_property, image_id),}


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x