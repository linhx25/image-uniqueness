#!/bin/bash
# scripts for uniqueness training and inference 
cd ~/Airbnb_unique/image-uniqueness

# inference on all NY data
init_state="./output/08-05_01:37:26/model.pt" # set your own trained model relative path
scene_dir="./output/data/debug_NY_idx.pkl" # must set this file to skip abnormal data in NY dataset 
bsub -M 30G -q gpu -gpu "num=5" -o inference.log \
python -m torch.distributed.launch --nproc_per_node=5 ./main_ddp.py --data_dir /export/projects2/szhang_text_project/Airbnb_Images_NYC/photos.hdf5 --init_state ${init_state} --scene_dir ${scene_dir} --n_epochs 0 --arch resnet18 --batch_size 256 --freeze_modules ~fc,avgpool,layer4 --freeze_first_n_epochs 40 --dataset AirbnbNYDataset