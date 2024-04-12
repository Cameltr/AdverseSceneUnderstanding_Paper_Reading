# 场景理解 域适应

This repo will keep updating ... 🤗

## UDA

### 训练方法

* (ISBI 2024) "When Visual Prompt Tuning Meets Source-Free
 Domain Adaptive Semantic Segmentation" [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2023/file/157c30da6a988e1cbef2095f7b9521db-Paper-Conference.pdf) [[code]](https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt)

* (ISBI 2024)"ConvLoRA and AdaBN Based Domain Adaptation via Self-Training" [[PDF]](https://arxiv.org/pdf/2402.04964.pdf) [[code]](https://github.com/aleemsidra/ConvLoRA)

* (AAAI 2024)"Parsing All Adverse Scenes Severity-Aware Semantic Segmentation with Mask-Enhanced Cross-Domain Consistency" [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/29251) [[code]](https://github.com/Cuzyoung/PASS)

* (ICCV 2023)"CMDA: Cross-Modality Domain Adaptation for Nighttime
Semantic Segmentation" [[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_CMDA_Cross-Modality_Domain_Adaptation_for_Nighttime_Semantic_Segmentation_ICCV_2023_paper.pdf) [[code]](https://github.com/XiaRho/CMDA)
  * P1: Conventional cameras fail to capture structural details and boundary information in low-light conditions. -> Event cameras -> unsupervised Cross-Modality Domain Adaptation (CMDA) framework to leverage multi-modality (Images and Events) information for nighttime semantic segmentation (数据不够，引入了额外的event信息，有点像先验)

* (ICCV 2023)"Contrastive Model Adaptation for Cross-Condition Robustness in Semantic Segmentation" [[PDF]](https://arxiv.org/pdf/2303.05194.pdf) [[code]](https://github.com/brdav/cma)
  * 利用参考图像的信息缩短域差异，进行信息交换和对比学习
  
* (ICCV 2023 Oral)"Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation"[[PDF]](https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/paper.pdf) [[code]](https://github.com/Red-Fairy/ZeroShotDayNightDA)
  * existing works rely heavily on domain knowledge derived from the task specific nighttime dataset. （训练数据依赖特定任务的数据集）-> zero-shot
  * image-level methods simply consider synthetic nighttime as pseudo-labeled data and overlook model-level feature extraction；model-level methods focus on adjusting model architecture but neglect image-level nighttime characteristics.（现有模型没有统筹考虑图像层面和模型层面的特点）-> 图像方面迁移后和原图相似性最小化，模型层面域适应最大化图像相似性

* (CVPR 2023)"MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation"[[pdf]](https://arxiv.org/pdf/2212.01322.pdf) [[code]](https://github.com/lhoyer/MIC)
    
    P1：Most previous UDA methods struggle with classes that have a similar visual appearance on the target domain as no ground truth is available to learn the slight appearance differences.（大多数以前的UDA方法都很难处理在目标域上具有相似视觉外观的类，因为没有基本事实可以用来学习轻微的外观差异。比如road/sidewalk，pedestrian/rider）->To address this problem, we propose to enhance UDA with spatial context relations as additional clues for robust visual recognition

* (CVPR 2023)"DiGA: Distil to Generalize and then Adapt for Domain Adaptive
Semantic Segmentation" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiGA_Distil_To_Generalize_and_Then_Adapt_for_Domain_Adaptive_CVPR_2023_paper.pdf)

* (CVPR 2023)"FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Truong_FREDOM_Fairness_Domain_Adaptation_Approach_to_Semantic_Scene_Understanding_CVPR_2023_paper.pdf) [[code]]()

* (CVPR 2023)"DA-DETR: Domain Adaptive Detection Transformer with Information Fusion"[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DA-DETR_Domain_Adaptive_Detection_Transformer_With_Information_Fusion_CVPR_2023_paper.pdf)
  
* (AAAI 2023 oral)"VBLC: Visibility Boosting and Logit-Constraint Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions" [[PDF]](https://arxiv.org/abs/2211.12256) [[code]](https://github.com/BIT-DA/VBLC)
     
    * P1: However, previous methods often reckon on additional reference images of the same scenes taken from normal conditions, which are quite tough to collect in reality. (训练数据的获取方式依赖参考)

    * P2：Furthermore, most of them mainly focus on individual adverse condition such as nighttime or foggy, weakening the model versatility when encountering other adverse weathers. (训练数据的主要关注夜晚和雾天，忽视其他恶劣场景)

    * P3: Second, for the self-training schemes prevailing in UDA, we observe the insufficient exploitation of predictions on unlabeled target samples for fear of overconfidence. (自训练方式过于自信，导致目标域样本利用不足)

* (NIPS 2022)"Unsupervised Domain Adaptation for Semantic Segmentation using Depth Distribution"[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/5c882988ce5fac487974ee4f415b96a9-Paper-Conference.pdf) [[code]](https://github.com/depdis/Depth_Distribution)

### 恶劣天气风格特征
* (ICCV 2023) "PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization"[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cho_PromptStyler_Prompt-driven_Style_Generation_for_Source-free_Domain_Generalization_ICCV_2023_paper.pdf)

* (arxiv 2023.5) "Condition-Invariant Semantic Segmentation" [[PDF]](https://arxiv.org/pdf/2305.17349.pdf) [[code]](https://github.com/SysCV/CISS)

* (ICCV 2023) "PODA: Prompt-driven Zero-shot Domain Adaptation"[[PDF]]( https://arxiv.org/pdf/2212.03241.pdf) [[code]](https://github.com/astra-vision/PODA)
  * New task：‘Prompt-driven Zero-shot Domain Adaptation’, where we adapt a model trained on a source domain using only a general description in natural language of the target domain

* (CVPR 2022) "FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation" [[PDF]](https://arxiv.org/pdf/2204.01587.pdf) [[code]](https://github.com/sohyun-l/fifo)

### 伪标签

* (CVPR 2023) "Learning Pseudo-Relations for Cross-domain Semantic Segmentation" [[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Learning_Pseudo-Relations_for_Cross-domain_Semantic_Segmentation_ICCV_2023_paper.pdf) [[code]]()

* (ICCV Workshop 2023) "SegDA: Maximum Separable Segment Mask with Pseudo Labels for Domain Adaptive Semantic Segmentation".  [[PDF]](https://arxiv.org/pdf/2308.05851) [[code]](https://github.com/SysCV/CISS)
  
* (CVPR 2023) "Continuous Pseudo-Label Rectified Domain Adaptive Semantic Segmentation
with Implicit Neural Representations" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gong_Continuous_Pseudo-Label_Rectified_Domain_Adaptive_Semantic_Segmentation_With_Implicit_Neural_CVPR_2023_paper.pdf) [[code]](https://github.com/ETHRuiGong/IR2F)

* (CVPR 2023) "UniDAformer: Unified Domain Adaptive Panoptic Segmentation Transformer
via Hierarchical Mask Calibration" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_UniDAformer_Unified_Domain_Adaptive_Panoptic_Segmentation_Transformer_via_Hierarchical_Mask_CVPR_2023_paper.pdf) [[code]]()

* (CVPR 2023) "Guiding Pseudo-labels with Uncertainty Estimation for Source-free Unsupervised Domain Adaptation"[[PDF]](https://arxiv.org/abs/2303.03770) [[code]](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation)
  
* (WACV 2023) "Refign: Align and Refine for Adaptation of Semantic Segmentation to Adverse Conditions" [[PDF]](https://arxiv.org/pdf/2207.06825.pdf) [[code]](https://github.com/brdav/refign)  
    *Ranks #1 on both the ACDC leaderboard—72.05 mIoU—and the Dark Zurich leaderboard—63.91 mIoU.*

     P1：A critical issue in this procedure is the error propagation of noisy labels, leading to a drift in pseudo-labels if unmitigated. 

### chain-of-domain

* (Arxiv 202403) "CoDA: Instructive Chain-of-Domain Adaptation with Severity-Aware Visual Prompt Tuning"[[PDF]](https://arxiv.org/abs/2403.17369) [[code]](https://github.com/Cuzyoung/CoDA)

### 数据生成

* (Arxiv 202403) "ZoDi: Zero-Shot Domain Adaptation with Diffusion-Based Image Transfer" [[PDF]](https://arxiv.org/pdf/2403.13652.pdf) 
  
* (Arxiv 202402) "ControlUDA: Controllable Diffusion-assisted Unsupervised Domain Adaptation
 for Cross-Weather Semantic Segmentation"[[PDF]](https://arxiv.org/pdf/2402.06446.pdf)

## One-shot UDA

* (CVPR 2023) "Informative Data Mining for One-shot Cross-Domain Semantic Segmentation" [[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Informative_Data_Mining_for_One-Shot_Cross-Domain_Semantic_Segmentation_ICCV_2023_paper.pdf) [[code]](https://github.com/yxiwang/IDM)

## Online UDA

* (CVPR 2024) "Memory-based Adapters for Online 3D Scene Perception" [[PDF]](https://arxiv.org/abs/2403.06974) [[code]](https://xuxw98.github.io/Online3D/)

* (ICCV 2023) "To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation"
[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Colomer_To_Adapt_or_Not_to_Adapt_Real-Time_Adaptation_for_Semantic_ICCV_2023_paper.pdf) [[code]](https://github.com/MarcBotet/hamlet)

* (ECCV 2022) "Online Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions"[[code]](https://github.com/theo2021/OnDA)

* (CVPR 2022) "On the Road to Online Adaptation for Semantic Image Segmentation" [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Volpi_On_the_Road_to_Online_Adaptation_for_Semantic_Image_Segmentation_CVPR_2022_paper.pdf)


## Test Time DA

* (AAAI 2024) "Exploring Sparse Visual Prompt for Domain Adaptive Dense Prediction" [[PDF]](https://arxiv.org/pdf/2312.12480.pdf) [[code]](https://github.com/Anonymous-012/SVDP)

* (CVPR 2024)"Continual-MAE: Adaptive Distribution Masked Autoencoders for Continual Test-Time Adaptation" [[PDF]](https://arxiv.org/pdf/2312.12480.pdf) 

* (arxiv 2404)"Test-Time Model Adaptation with Only Forward Passes" [[PDF]](https://arxiv.org/pdf/2404.01650.pdf) 
  
* (arxiv 2402)"BECoTTA: Input-dependent Online Blending of Experts for Continual Test-time Adaptation" [[PDF]](https://arxiv.org/pdf/2402.08712.pdf) [[code]](https://becotta-ctta.github.io/)

* (CVPR 2023)"Robust Test-Time Adaptation in Dynamic Scenarios" [[PDF]](https://arxiv.org/abs/2303.13899) [[code]](https://github.com/BIT-DA/RoTTA)
  
* (CVPR 2023)"Dynamically Instance-Guided Adaptation: A Backward-free Approach for Test-Time Domain Adaptive Semantic Segmentation" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Dynamically_Instance-Guided_Adaptation_A_Backward-Free_Approach_for_Test-Time_Domain_Adaptive_CVPR_2023_paper.pdf) [[code]](https://github.com/Waybaba/DIGA)
  


## Domain Generalization
* (CVPR 2024) "Collaborating Foundation models for Domain Generalized Semantic Segmentation"[[PDF]](https://arxiv.org/abs/2312.09788) [[code]](https://github.com/yasserben/CLOUDS)

* (CVPR 2024) "Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation"[[PDF]](https://arxiv.org/pdf/2312.04265.pdf) [[code]](https://github.com/w1oves/Rein)

* (Arxiv 202312) "DGInStyle: Domain-Generalizable Semantic Segmentation with Image Diffusion Models and Stylized Semantic Control"[[PDF]](https://arxiv.org/abs/2312.03048) [[code]](https://dginstyle.github.io/)
  
* (CVPR 2024)"Style Blind Domain Generalized Semantic Segmentation via Covariance Alignment and Semantic Consistence Contrastive Learning" [[PDF]](https://arxiv.org/pdf/2403.06122.pdf) [[code]](https://github.com/root0yang/BlindNet)

* (CVPR 2023) "CLIP the Gap: A Single Domain Generalization Approach for Object Detection" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Vidit_CLIP_the_Gap_A_Single_Domain_Generalization_Approach_for_Object_CVPR_2023_paper.pdf) [[code]](https://github.com/vidit09/domaingen)

* (NIPS 2022)"Adversarial Style Augmentation for Domain Generalized Urban-Scene Segmentation" [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/023d94f44110b9a3c62329beec739772-Paper-Conference.pdf) 

## Anomaly Detection
* (ICCV 2023 oral)"Unmasking Anomalies in Road-Scene Segmentation" [[PDF]](https://arxiv.org/abs/2307.13316) [[code]](https://github.com/shyam671/Mask2Anomaly-Unmasking-Anomalies-in-Road-Scene-Segmentation)
  
* (ICCV 2023)"RbA: Segmenting Unknown Regions Rejected by All" [[PDF]](https://arxiv.org/abs/2211.14293) [[code]](https://github.com/NazirNayal8/RbA)
  
* (NIPS 2023)"ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation" [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2023/file/8dcc306a2522c60a78f047ab8739e631-Paper-Conference.pdf) [[code]](https://github.com/gaozhitong/ATTA)


## 其他相关


* (CVPR 2023)"StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning" [[PDF]](https://arxiv.org/pdf/2302.09309.pdf) [[code]](https://github.com/lovelyqian/StyleAdv-CDFSL)


* (ICCV 2023 Oral) "Iterative Prompt Learning for Unsupervised Backlit Image Enhancement" [[PDF]](https://browse.arxiv.org/pdf/2303.17569.pdf) [[code]](https://github.com/ZhexinLiang/CLIP-LIT)
  
* (Arxiv 2023) "MoWE: Mixture of Weather Experts for Multiple Adverse Weather Removal".  [[PDF]](https://arxiv.org/pdf/2303.13739.pdf)
  
* (CVPR 2023) "Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions" [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf)




# Dataset

* FoggyCityscape
* FoggyDriving
* FoggyZurich
* ACDC
* BDD100K
* DarkZurich
* Nighttime Driving
* ACG (Adverse-Condition Generalization):o 121 fog, 225 rain, 276 snow, and 300 night images [[link]](https://github.com/brdav/cma)
* DSEC Night_Semantic: 5 nighttime sequences of Zurich City 09a-e, and includes 1,692 training samples and 150 testing samples. [[link]](https://dsec.ifi.uzh.ch/dsec-datasets/download/)
* [MUAD Dataset](https://muad-dataset.github.io/) a synthetic dataset for autonomous driving with multiple uncertainty types and tasks. It contains 10413 in total: 3420 images in the train set, 492 in the validation set and 6501 in the test set. The test set is divided as follows: 551 in the normal set, 102 in the normal set no shadow, 1668 in the OOD set, 605 in the low adversity set and 602 images in the high adversity set 1552 in the low adversity with OOD set and 1421 images in the high adversity with OOD set. All of these sets cover day and night conditions, with 2/3 being day images and 1/3 night images. Test datasets address diverse weather conditions (rain, snow, and fog with two different intensity levels) and multiple OOD objects. [[link1]](https://drive.google.com/drive/folders/1CJaM1hdjZr9RMQVB4JFJ7QMUFIHYxFVo?usp=sharing) 
 [[link2]](https://we.tl/t-5ZmFYzaeVT)


video dataset

[KITTI](https://www.cvlibs.net/datasets/kitti/eval_step.php)

[SHIFT](https://dl.cv.ethz.ch/shift)

[Rainy-foggy cityscape](https://team.inria.fr/rits/computer-vision/weather-augment/)
