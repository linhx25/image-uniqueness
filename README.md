# Image-uniqueness
This repository contains the code and resources for a research project on modeling image uniqueness using Airbnb data. The core of this research is a method that leverages contrastive learning to determine how unique an image is.

## Method Overview
The primary approach uses contrastive learning to measure uniqueness. The central idea is that an image that is unique will be less similar to its counterparts. Consequently, the InfoNCE loss serves as an effective measure of uniqueness, as it quantifies how different an image is from others.

The training and evaluation infrastructure for this project can be found in ``main.py`` and ``main_ddp.py`` (for distributed training). The complete procedure can be found in ``run.sh``.

## Visualization & Analysis
I use [RELAX: Representation Learning Explainability](https://arxiv.org/abs/2112.10161), a gradient-based importance visualization method, to further analyze the image representations. The Jupyter notebook in this repository provides a detailed analysis of the images using this method.


