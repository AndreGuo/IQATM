# Deep **T**one **M**apping Opetator Using **I**mage **Q**uality **A**ssessment Inspired Semi-supervised Training (IQATM)
This is the official Tensorflow implementation of paper  
[C. Guo and X. Jiang, "Deep **T**one-**M**apping Operator Using **I**mage **Q**uality **A**ssessment Inspired Semi-Supervised Learning," in IEEE Access, vol. 9, pp. 73873-73889, 2021](https://ieeexplore.ieee.org/document/9431092).  
## Introduction
Tone-mapping is to display HDR (High Dynamic Range) image on a traditional SDR (Standrad Dynamic Range, a.k.a. Low Dynamic Range, LDR) display, its result is usually stored as SDR image. Scilicet, tone-mapping is the reverse prosess of single-shot HDR image generation (a.k.a. reverse/inverse tone-mapping).  
There're 2 types of HDR content: photometrically *linear* one which is used in photograhpy, medicine and image based lighting, and *non-linear* one usually transformed by curve like PQ/HLG and used in film and television. Specifically, our work deals with **linear** HDR content.
## Prerequisites
1. Unbuntu with PyCharm IDE
2. Python 2.7
3. NVIDIA GPU & CUDA CuDNN (CUDA 8.0)
4. Tensorflow-GPU 1.x TODO
5. Other packages: opencv-python, imageio, easydict, etc.
## How to test
### 0. Downloading checkpoint?
TODO
### 1. Preparing data
Place your testing HDR images under `/dataset/test` floder. We recommend to use `.hdr` encapsulation, otherwise you would like to go to `/utils/configs.py` and change `config.data.appendix_hdr` to your one as long as package `imageio` support.
### 2. Generating TFRecord
Run `/generate_tfrec.py`, (optional) you can set `test_resize_to_half = True` if your GPU is out of memory.
### 3. Testing
Run `/test.py`, results will be stored under `/result` floder.
## How to Train
TODO
## Acknowledgments
Our Tensorflow code was designed based on [Deep Reformulated Laplacian Tone Mapping (DRLTM)](https://github.com/linmc86/Deep-Reformulated-Laplacian-Tone-Mapping), this greatly simplified our coding since we are not experienced in computer science.
