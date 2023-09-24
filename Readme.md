# 恶劣场景理解 无监督语义分割

This repo will keep updating ... 🤗

## 训练方式




* "MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation"(CVPR 2023)[[pdf]](https://arxiv.org/pdf/2212.01322.pdf) [[code]](https://github.com/lhoyer/MIC)

* "Refign: Align and Refine for Adaptation of Semantic Segmentation to Adverse Conditions"(WACV 2023) [[PDF]](https://arxiv.org/pdf/2207.06825.pdf) [[code]](https://github.com/brdav/refign)  
    *Ranks #1 on both the ACDC leaderboard—72.05 mIoU—and the Dark Zurich leaderboard—63.91 mIoU.*

     P1：过去方法的伪标签噪声很多 

* "VBLC: Visibility Boosting and Logit-Constraint Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions"(AAAI 2023 oral) [[PDF]](https://arxiv.org/abs/2211.12256) [[code]](https://github.com/BIT-DA/VBLC)
     
     P1: However, previous methods often reckon on additional reference images of the same scenes taken from normal conditions, which are quite tough to collect in reality. (训练数据的获取方式依赖参考)
     
     P2：Furthermore, most of them mainly focus on individual adverse condition such as nighttime or foggy, weakening the model versatility when encountering other adverse weathers. (训练数据的主要关注夜晚和雾天，忽视其他恶劣场景)
     
     P3: Second, for the self-training schemes prevailing in UDA, we observe the insufficient exploitation of predictions on unlabeled target samples for fear of overconfidence. (自训练方式过于自信，导致目标域样本利用不足)

## 雾天特征
* "FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation"(CVPR 2022) [[PDF]](https://arxiv.org/pdf/2204.01587.pdf) [[code]](https://github.com/sohyun-l/fifo)


## 伪标签

* "SegDA: Maximum Separable Segment Mask with Pseudo Labels for Domain Adaptive Semantic Segmentation". (ICCV Workshop 2023) [[PDF]](https://arxiv.org/pdf/2308.05851) 
  


## 其他相关
  
* "MoWE: Mixture of Weather Experts for Multiple Adverse Weather Removal". (arxiv 2023) [[PDF]](https://arxiv.org/pdf/2303.13739.pdf)

* "EDAPS: Enhanced Domain-Adaptive Panoptic Segmentation"(ICCV 2023)[[PDF]](https://arxiv.org/pdf/2304.14291.pdf)[[code]](https://github.com/susaha/edaps)
  






# Dataset

* FoggyCityscape
* FoggyDriving
* FoggyZurich
* ACDC
* BDD100K
