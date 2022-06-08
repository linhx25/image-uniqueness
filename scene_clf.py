import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.nn import functional as F
from operator import itemgetter
import sys; sys.path.append("../")
import src.dataset

import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

### The GPU output is slightly different from CPU. I don't konw why.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def hook_feature(module, input, output):
    global features_blobs
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    
    return model


if __name__ == "__main__":
    inpath = "/export/projects2/szhang_text_project/Airbnb_unique/photo_library/photo_library_batch0.h5"
    score_path="/export/home/rcsguest/rcs_hlin/Airbnb_unique/image-uniqueness/output/05-26_15:32:10/train_loss.pkl"
    
    ## dataset
    ds = src.dataset.AirbnbDataset(inpath)
    loader = DataLoader(ds, batch_size=64, drop_last=False)
    mapping = ds.idx_mapping.reset_index().set_index("image_id")
    
    ## uniqueness
    uniqueness = pd.read_pickle(score_path)
    
    ## labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()
    
    ## model
    features_blobs = []
    model = load_model()
    model.to(device)
    
    ## transform
    tsfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader.dataset.transform = tsfm

    ## get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    preds = []
    for batch in tqdm(loader):
        images, ids = batch["image"], batch["label"]
        
        ## prediction
        with torch.no_grad():
            logit = model(images.to(device))
            h_x = F.softmax(logit, 1).data.squeeze()
            prob, idx = h_x.sort(-1, True)
            prob = prob.cpu().numpy()[:, 0]
            idx = idx.cpu().numpy()

        ## output the IO prediction
        io_pred = labels_IO[idx[:,:10]].mean(axis=-1) # vote for the indoor or outdoor
        io_pred = (io_pred > 0.5).astype(int)

        ## output the prediction of scene category
        cate_pred = itemgetter(*idx[:, 0])(classes)

        ## for our forcast
        preds.append(pd.DataFrame({
            "in/out": io_pred, 
            "scene": cate_pred,
            "prob": prob,
        }, index=ids))

    
    preds = pd.concat(preds).reindex(uniqueness.index)
    uniqueness[["in/out", "scene", "prob"]] = preds
    uniqueness.to_pickle("/export/home/rcsguest/rcs_hlin/Airbnb_unique/image-uniqueness/output/05-26_15:32:10/uniqueness.pkl")

        