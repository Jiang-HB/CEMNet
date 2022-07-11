# Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration (ICCV2021)

PyTorch implementation of CEMNet for ICCV'2021 paper ["Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration"](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Sampling_Network_Guided_Cross-Entropy_Method_for_Unsupervised_Point_Cloud_Registration_ICCV_2021_paper.pdf), by *Haobo Jiang, Yaqi Shen,  Jin Xie, Jun Li, Jianjun Qian, and Jian Yang* from PCA Lab, Nanjing University of Science and Technology, China. 

This paper focuses on unsupervised deep learning for 3D point clouds registration. If you find this project useful, please cite:

```bash
@inproceedings{jiang2021sampling,
title={{S}ampling {N}etwork {G}uided {C}ross-{E}ntropy {M}ethod for {U}nsupervised {P}oint {C}loud {R}egistration},
author={Jiang, Haobo and Shen, Yaqi and Xie, Jin and Li, Jun and Qian, Jianjun and Yang, Jian},
booktitle={ICCV},
year={2021}
}
```

## Introduction

In this paper, by modeling the point cloud registration task as a Markov decision process, we propose an end-to-end deep model embedded with the cross-entropy method (CEM) for unsupervised 3D registration.
Our model consists of a sampling network  module and a differentiable CEM module. In our sampling network module, given a pair of point clouds, the sampling network learns a prior sampling distribution over the transformation space. The learned sampling distribution can be used as a "good" initialization of the differentiable CEM module. In our differentiable CEM module, we first propose a maximum consensus criterion based alignment metric as the reward function for the point cloud registration task. Based on the reward function, for each state, we then construct a fused score function to evaluate the sampled transformations, where we weight the current and future rewards of the transformations. Particularly, the future rewards of the sampled transforms are obtained by performing the iterative closest point (ICP) algorithm  on the transformed state. Extensive experimental results demonstrate the good registration performance of our method on benchmark datasets. 

## Requirements

Before running our code, you need to install the `cemlib` and `batch_svd` libraries via:

```bash
bash run.sh
```
(If you meet something error when intall `batch_svd`, please refer to [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd).)

## Dataset Preprocessing

We generated the used dataset files `modelnet40_normal_n2048.pth` , `7scene_normal_n2048.pth` and `icl_nuim_normal_n2048.pth` by preprocessing the raw point clouds of *ModelNet40*, *7Scene* and *ICL-NUIM* , and uploaded them to [GoogleDisk](https://drive.google.com/drive/folders/1ne9naYI8M8v4Lv0L9AcQm60Jqb8ciQ6t?usp=sharing). Also, you can generate them by yourself via:

```bash
cd datasets
python3 process_dataset.py
```

After that, you need modify the dataset paths in `utils/options.py`. 

## Pretrained Model

We uploaded the pretrained models as below: 

*ModelNe40*: `results/modelnet40_n768_unseen0_noise0_seed123/model.pth`,

*7Scene* : `results/scene7_n768_unseen0_noise0_seed123/model.pth`, 

*ICL-NUIM*: `results/icl_nuim_n768_unseen0_noise0_seed123/model.pth`.

## Acknowledgments
We thank the authors of 
- [DCP](https://github.com/WangYueFt/dcp)
- [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)

for open sourcing their methods.
