# PySOT

**PySOT** is a software system designed by SenseTime Video Intelligence Research team. It implements state-of-the-art single object tracking algorithms, including [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) and [SiamMask](https://arxiv.org/abs/1812.05050). It is written in Python and powered by the [PyTorch](https://pytorch.org) deep learning framework. This project also contains a Python port of toolkit for evaluating trackers.

PySOT has enabled research projects, including: [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html), [DaSiamRPN](https://arxiv.org/abs/1808.06048), [SiamRPN++](https://arxiv.org/abs/1812.11703), and [SiamMask](https://arxiv.org/abs/1812.05050).

This repository additionally added [SiamCar](https://arxiv.org/abs/1911.07241v2)

## Introduction

The goal of PySOT is to provide a high-quality, high-performance codebase for visual tracking *research*. It is designed to be flexible in order to support rapid implementation and evaluation of novel research. PySOT includes implementations of the following visual tracking algorithms:

- [SiamCar](https://arxiv.org/abs/1911.07241v2)
- [SiamMask](https://arxiv.org/abs/1812.05050)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [SiamFC](https://arxiv.org/abs/1606.09549)

using the following backbone network architectures:

- [ResNet{18, 34, 50}](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

Evaluation toolkit can support the following datasets:

:paperclip: [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 
:paperclip: [VOT16/18/19](http://votchallenge.net) 
:paperclip: [VOT18-LT](http://votchallenge.net/vot2018/index.html) 
:paperclip: [LaSOT](https://arxiv.org/pdf/1809.07845.pdf) 
:paperclip: [UAV123](https://arxiv.org/pdf/1804.00518.pdf)

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [PySOT Model Zoo](MODEL_ZOO.md). SiamCar baseline results and trained models available for download here:

[general_model](https://pan.baidu.com/s/1kIbKxCu1O3PXt9wQik4EVQ) code: xjpz  
[got10k_model](https://pan.baidu.com/s/1KSVgaz5KYP2Ar2DptnfyGQ) code: p4zx  
[LaSOT_model](https://pan.baidu.com/s/1g15wGSq-LoZUBxYQwXCP6w) code: 6wer

[Google Drive Link](https://drive.google.com/drive/folders/1yiFJNv5JLBvcdw63nw3aUedoXt7pDGij?usp=sharing)
and put them into the respected `pysot/experiments` directory. 

## Installation

Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).

## Conda Installation
You can creating an conda environment from an ``environment.yml`` file
```
conda env create -f environment.yml
```
The first line of the ``yml`` file sets the new environment's name. For more details see [Conda Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

Activate the new environment
```
conda activate pysot
```
Verify the new environment
```
conda env list
```

## Quick Start: Using PySOT

### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

### Download models
Download models in [PySOT Model Zoo](MODEL_ZOO.md) and put the model.pth in the correct directory in experiments

### Webcam demo
```bash
python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Produce Prediction Results for Multiple Videos

```
python tools/eval_benchmark.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth  \
    --video_folder data_dir \
    --gt groundtruth_file_name

```
Prediction results with same format `<x>, <y>, <w>, <h>` will be saved in `results/` directory or other directory specified with `-save_dir` flag.

Arguments:
`--config`: Path to a pretrained model configuration file.
`--snapshot`: Path to pretrained model weights
`--video_folder`: Path to the directory that contains all the videos.
`--save_dir`: Directory to save the prediction results. Default is `pygoturn/result` directory
`--gt`: Ground truth file name such as groundtruth.txt, groundtruth_rect.txt or gt.txt

All the tracking sequences need to be at the same level as follow
```
./data
   ├── VideoA                         # Arbitrary sequence name
   │   ├── img                        # Directory that has all the frames
   │   ├── groundtruth.txt            # Txt file of ground truth
   ├── VideoB                         # Arbitrary sequence name
   │   ├── img                        # Directory that has all the frames
   │   ├── groundtruth.txt            # Txt file of ground truth
   ├── VideoC                         # Arbitrary sequence name
   │   ├── img                        # Directory that has all the frames
   │   ├── groundtruth.txt            # Txt file of ground truth
```
To move sub directories up a level, you can do the following:

1. Verify finding the sub directories correctly
```
find . -maxdepth 2  -print
```

2. Move the sub directories up
```
find . -maxdepth 2  -print -exec mv {} . \;
```

The results can be evaluated via [track_eval](https://github.com/versey-sherry/track_eval)

### Download testing datasets
Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker
```bash
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```
The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker
assume still in experiments/siamrpn_r50_l234_dwxcorr_8gpu
``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:
See [TRAIN.md](TRAIN.md) for detailed instruction.


### Getting Help :hammer:
If you meet problem, try searching our GitHub issues first. We intend the issues page to be a forum in which the community collectively troubleshoots problems. But please do **not** post **duplicate** issues. If you have similar issue that has been closed, you can reopen it.

- `ModuleNotFoundError: No module named 'pysot'`

:dart:Solution: Run `export PYTHONPATH=path/to/pysot` first before you run the code.

- `ImportError: cannot import name region`

:dart:Solution: Build `region` by `python setup.py build_ext —-inplace` as decribled in [INSTALL.md](INSTALL.md).


## References

- [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/abs/1812.05050).
  Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://arxiv.org/abs/1812.11703).
  Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [Distractor-aware Siamese Networks for Visual Object Tracking](https://arxiv.org/abs/1808.06048).
  Zheng Zhu, Qiang Wang, Bo Li, Wu Wei, Junjie Yan, Weiming Hu.
  The European Conference on Computer Vision (ECCV), 2018.

- [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html).
  Bo Li, Wei Wu, Zheng Zhu, Junjie Yan, Xiaolin Hu.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

- [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549).
  Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr.
  The European Conference on Computer Vision (ECCV) Workshops, 2016.
  
## Contributors

- [Fangyi Zhang](https://github.com/StrangerZhang)
- [Qiang Wang](http://www.robots.ox.ac.uk/~qwang/)
- [Bo Li](http://bo-li.info/)
- [Zhiyuan Chen](https://zyc.ai/)

## License

PySOT is released under the [Apache 2.0 license](https://github.com/STVIR/pysot/blob/master/LICENSE). 
