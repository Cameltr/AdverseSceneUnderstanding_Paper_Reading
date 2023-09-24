# 恶劣场景理解 无监督语义分割

This repo will keep updating ... 🤗

## UDA训练方式

* (AAAI 2023 oral)"VBLC: Visibility Boosting and Logit-Constraint Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions" [[PDF]](https://arxiv.org/abs/2211.12256) [[code]](https://github.com/BIT-DA/VBLC)
     
     P1: However, previous methods often reckon on additional reference images of the same scenes taken from normal conditions, which are quite tough to collect in reality. (训练数据的获取方式依赖参考)
     
     P2：Furthermore, most of them mainly focus on individual adverse condition such as nighttime or foggy, weakening the model versatility when encountering other adverse weathers. (训练数据的主要关注夜晚和雾天，忽视其他恶劣场景)
     
     P3: Second, for the self-training schemes prevailing in UDA, we observe the insufficient exploitation of predictions on unlabeled target samples for fear of overconfidence. (自训练方式过于自信，导致目标域样本利用不足)

* (CVPR 2023)"MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation"[[pdf]](https://arxiv.org/pdf/2212.01322.pdf) [[code]](https://github.com/lhoyer/MIC)
    
    P1：Most previous UDA methods struggle with classes that have a similar visual appearance on the target domain as no ground truth is available to learn the slight appearance differences.（大多数以前的UDA方法都很难处理在目标域上具有相似视觉外观的类，因为没有基本事实可以用来学习轻微的外观差异。比如road/sidewalk，pedestrian/rider）->To address this problem, we propose to enhance UDA with spatial context relations as additional clues for robust visual recognition

    利用masked图像的上下文线索信息来增加鲁棒性


* (CVPR 2023)"DiGA: Distil to Generalize and then Adapt for Domain Adaptive
Semantic Segmentation" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiGA_Distil_To_Generalize_and_Then_Adapt_for_Domain_Adaptive_CVPR_2023_paper.pdf)

* (CVPR 2023)"FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Truong_FREDOM_Fairness_Domain_Adaptation_Approach_to_Semantic_Scene_Understanding_CVPR_2023_paper.pdf) [[code]]()

* (CVPR 2023)"DA-DETR: Domain Adaptive Detection Transformer with Information Fusion"[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DA-DETR_Domain_Adaptive_Detection_Transformer_With_Information_Fusion_CVPR_2023_paper.pdf)

* (NIPS 2022)"Unsupervised Domain Adaptation for Semantic Segmentation using Depth Distribution"[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/5c882988ce5fac487974ee4f415b96a9-Paper-Conference.pdf) [[code]](https://github.com/depdis/Depth_Distribution)

* (ICCV 2023)"Contrastive Model Adaptation for Cross-Condition Robustness in Semantic Segmentation" [[PDF]](https://arxiv.org/pdf/2303.05194.pdf) [[code]](https://github.com/brdav/cma)
* (ICCV 2023)"To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation" [[PDF]](https://arxiv.org/pdf/2307.15063.pdf) [[code]]( https://github.com/MarcBotet/hamlet)
* (ICCV 2023)"PODA: Prompt-driven Zero-shot Domain Adaptation"[[PDF]]( https://arxiv.org/pdf/2212.03241.pdf) [[code]](https://github.com/astra-vision/PODA)
* (ICCV 2023 Oral)"Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation"[[PDF]](https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/paper.pdf) [[code]](https://github.com/Red-Fairy/ZeroShotDayNightDA)



## 雾天特征
* (CVPR 2022) "FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation" [[PDF]](https://arxiv.org/pdf/2204.01587.pdf) [[code]](https://github.com/sohyun-l/fifo)


## 伪标签

* (ICCV Workshop 2023) "SegDA: Maximum Separable Segment Mask with Pseudo Labels for Domain Adaptive Semantic Segmentation".  [[PDF]](https://arxiv.org/pdf/2308.05851) 
  
* (CVPR 2023) "Continuous Pseudo-Label Rectified Domain Adaptive Semantic Segmentation
with Implicit Neural Representations" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gong_Continuous_Pseudo-Label_Rectified_Domain_Adaptive_Semantic_Segmentation_With_Implicit_Neural_CVPR_2023_paper.pdf) [[code]](https://github.com/ETHRuiGong/IR2F)

* (WACV 2023) "Refign: Align and Refine for Adaptation of Semantic Segmentation to Adverse Conditions" [[PDF]](https://arxiv.org/pdf/2207.06825.pdf) [[code]](https://github.com/brdav/refign)  
    *Ranks #1 on both the ACDC leaderboard—72.05 mIoU—and the Dark Zurich leaderboard—63.91 mIoU.*

     P1：A critical issue in this procedure is the error propagation of noisy labels, leading to a drift in pseudo-labels if unmitigated. 

* (CVPR 2023) "UniDAformer: Unified Domain Adaptive Panoptic Segmentation Transformer
via Hierarchical Mask Calibration" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_UniDAformer_Unified_Domain_Adaptive_Panoptic_Segmentation_Transformer_via_Hierarchical_Mask_CVPR_2023_paper.pdf) [[code]]()

* (CVPR 2023) "Guiding Pseudo-labels with Uncertainty Estimation for Source-free Unsupervised Domain Adaptation"[[PDF]](https://arxiv.org/abs/2303.03770) [[code]](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation)
  


## 其他相关
  
* (Arxiv 2023) "MoWE: Mixture of Weather Experts for Multiple Adverse Weather Removal".  [[PDF]](https://arxiv.org/pdf/2303.13739.pdf)

* (ICCV 2023) "EDAPS: Enhanced Domain-Adaptive Panoptic Segmentation" [[PDF]](https://arxiv.org/pdf/2304.14291.pdf)[[code]](https://github.com/susaha/edaps)
  
* (CVPR 2023) "Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf)




# Dataset

* FoggyCityscape
* FoggyDriving
* FoggyZurich
* ACDC
* BDD100K
