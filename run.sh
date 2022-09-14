#!/bin/bash
# scripts for uniqueness training and inference 

scene_dir="./output/data/debug_NY_idx.pkl" # must set this file to skip abnormal data in NY dataset 
data_dir="/export/projects2/szhang_text_project/Airbnb_Images_NYC/photos.hdf5"

# 1. train on all NY data (including inference)
bsub -M 30G -q gpu -gpu "num=5" -o train.log \
python -m torch.distributed.launch --nproc_per_node=5 ./main_ddp.py --data_dir ${data_dir} --n_epochs 40 --arch resnet18 --pretrained --batch_size 256 --freeze_modules "~fc,avgpool,layer4"  --freeze_first_n_epochs 40 --dataset AirbnbNYDataset --scene_dir ${scene_dir}

# 2. scene classification based on step 1
bsub -M 50G -o scene.log -q gpu -gpu "num=1" \
python ./scene_clf.py --data_dir ${data_dir} --outdir "../output/08-05_01:37:26/" --dataset AirbnbNYDataset --scene_dir ${scene_dir}

# inference alone on all NY data
init_state="./output/08-05_01:37:26/model.pt" # set your own trained model relative path
bsub -M 30G -q gpu -gpu "num=5" -o inference.log \
python -m torch.distributed.launch --nproc_per_node=5 ./main_ddp.py --data_dir ${data_dir} --init_state ${init_state} --scene_dir ${scene_dir} --n_epochs 0 --arch resnet18 --batch_size 256 --freeze_modules ~fc,avgpool,layer4 --freeze_first_n_epochs 40 --dataset AirbnbNYDataset

# inference alone on all NY data + PS data (in the data_dir)
bsub -M 30G -q gpu -gpu "num=5" -o inference.log \
python -m torch.distributed.launch --nproc_per_node=5 ./main_ddp.py --data_dir "/export/projects2/szhang_text_project/Airbnb_Images_NYC/" --init_state ${init_state} --scene_dir ${scene_dir} --n_epochs 0 --arch resnet18 --batch_size 256 --freeze_modules ~fc,avgpool,layer4  --freeze_first_n_epochs 40 --dataset AirbnbNYDataset