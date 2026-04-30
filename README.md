# HYRET-CHANGE: A HYBRID RETENTIVE NETWORK FOR REMOTE SENSING CHANGE DETECTION

---

## 👥 Contributors

This repository was extended as part of **[CSC 722]** at **[NCSU]**.

| Team Members |
|------|
| [Janelle Correia] (jlcorrei) |
| [Abishek Viswanath Pittamandalam] (apittam) |
| [Lawrence Stephenson] (lasteph2)|

**Instructor:** [Dr. Raju Vatsavai]

---

## 🔬 Extended Study: Design Choice Analysis

This fork extends the original HyRet-Change implementation to systematically study the impact of fundamental design choices on change detection performance, inspired by ["Be the Change You Want to See: Revisiting Remote Sensing Change Detection Practices"](https://arxiv.org/pdf/2507.03367).

We evaluate the impact of:
- **Backbone architecture** (ResNet-50, ResNet-101, Swin-T, Swin-B, MambaVision-S-1k)
- **Backbone pretraining** (ImageNet-1k, ImageNet-22k, Cityscapes Semantic Segmentation)
- **Data augmentation** (horizontal/vertical flip, rotation, crop at 30% probability)
- **Loss functions** (Cross-Entropy, Dice, CE+Dice)

---

## 📦 Datasets

- [LEVIR-CD](https://justchenhao.github.io/LEVIR/) — from the [paper](https://doi.org/10.3390/rs12101662)
- [WHU-CD](https://gpcv.whu.edu.cn/data/building_dataset.html) — from the [paper](https://doi.org/10.1109/TGRS.2018.2858817)
- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) — from the [paper](https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf)

---

## 📄 Papers

- [HyRet-Change: A hybrid retentive network for remote sensing change detection](https://arxiv.org/pdf/2506.12836) — [HuggingFace](https://huggingface.co/mustansarfiaz/HyRet)
- [Be the Change You Want to See: Revisiting Remote Sensing Change Detection Practices](https://arxiv.org/pdf/2507.03367) — [HuggingFace](https://huggingface.co/collections/blaz-r/be-the-change-btc)

---

## 🚀 Quick Start
- Install requirements:
    - Run ```bash setup.sh```
    - In a new terminal, activate the environment via ```conda activate hyret```
    - Enter the project directory: ```cd HyRect-Change```
Once these steps are complete, proceed to the evaluation commands listed below.

---

## Commands to Reproduce Results

### LEVIR-CD

```bash
# ResNet-50 (Baseline)
python eval_cd.py --data_name LEVIR --project_name hyret_levir --checkpoint_name hyret_levir_ckpt.pt

# ResNet-101
python eval_cd.py --data_name LEVIR --project_name hyret_levir_resnet101 --checkpoint_name hyret_levir_resnet101_ckpt.pt

# Swin-Tiny
python eval_cd.py --data_name LEVIR --project_name hyret_levir_swint --checkpoint_name hyret_levir_swint_ckpt.pt

# Swin-Tiny + Cityscapes Semantic Segmentation
python eval_cd.py --data_name LEVIR --project_name hyret_levir_swint_citysem --checkpoint_name hyret_levir_swint_citysem_ckpt.pt

# Swin-Base
python eval_cd.py --data_name LEVIR --project_name hyret_levir_swinb --checkpoint_name hyret_levir_swinb_ckpt.pt

# Swin-Base + Cityscapes Semantic Segmentation
python eval_cd.py --data_name LEVIR --project_name hyret_levir_swinb_citysem --checkpoint_name hyret_levir_swinb_citysem_ckpt.pt

# Mamba1k
python eval_cd.py --data_name LEVIR --project_name hyret_levir_mamba1k --checkpoint_name hyret_levir_mamba1k_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation
python eval_cd.py --data_name LEVIR --project_name hyret_levir_resnet50_flip30 --checkpoint_name hyret_levir_resnet50_flip30_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation + 30% Crop
python eval_cd.py --data_name LEVIR --project_name hyret_levir_resnet50_flip30_crop --checkpoint_name hyret_levir_resnet50_flip30_crop_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation + 30% Crop + CE+Dice Loss
python eval_cd.py --data_name LEVIR --project_name hyret_levir_resnet50_flip30_crop_ce_dice --checkpoint_name hyret_levir_resnet50_flip30_crop_ce_dice_ckpt.pt
```

### CDD

```bash
# ResNet-50 (Baseline)
python eval_cd.py --data_name CDD --project_name hyret_cdd --checkpoint_name hyret_cdd_ckpt.pt

# ResNet-101
python eval_cd.py --data_name CDD --project_name hyret_cdd_resnet101 --checkpoint_name hyret_cdd_resnet101_ckpt.pt

# Swin-Tiny
python eval_cd.py --data_name CDD --project_name hyret_cdd_swint --checkpoint_name hyret_cdd_swint_ckpt.pt

# Swin-Tiny + Cityscapes Semantic Segmentation
python eval_cd.py --data_name CDD --project_name hyret_cdd_swint_citysem --checkpoint_name hyret_cdd_swint_citysem_ckpt.pt

# Swin-Base
python eval_cd.py --data_name CDD --project_name hyret_cdd_swinb --checkpoint_name hyret_cdd_swinb_ckpt.pt

# Swin-Base + Cityscapes Semantic Segmentation
python eval_cd.py --data_name CDD --project_name hyret_cdd_swinb_citysem --checkpoint_name hyret_cdd_swinb_citysem_ckpt.pt

# Mamba1k
python eval_cd.py --data_name CDD --project_name hyret_cdd_mamba1k --checkpoint_name hyret_cdd_mamba1k_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation
python eval_cd.py --data_name CDD --project_name hyret_cdd_resnet50_flip30 --checkpoint_name hyret_cdd_resnet50_flip30_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation + 30% Crop
python eval_cd.py --data_name CDD --project_name hyret_cdd_resnet50_flip30_crop --checkpoint_name hyret_cdd_resnet50_flip30_crop_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation + 30% Crop + CE+Dice Loss
python eval_cd.py --data_name CDD --project_name hyret_cdd_resnet50_flip30_crop_ce_dice --checkpoint_name hyret_cdd_resnet50_flip30_crop_ce_dice_ckpt.pt
```

### WHU-CD

```bash
# ResNet-50 (Baseline)
python eval_cd.py --data_name WHU --project_name hyret_whu --checkpoint_name hyret_whu_ckpt.pt

# ResNet-101
python eval_cd.py --data_name WHU --project_name hyret_whu_resnet101 --checkpoint_name hyret_whu_resnet101_ckpt.pt

# Swin-Tiny
python eval_cd.py --data_name WHU --project_name hyret_whu_swint --checkpoint_name hyret_whu_swint_ckpt.pt

# Swin-Tiny + Cityscapes Semantic Segmentation
python eval_cd.py --data_name WHU --project_name hyret_whu_swint_citysem --checkpoint_name hyret_whu_swint_citysem_ckpt.pt

# Swin-Base
python eval_cd.py --data_name WHU --project_name hyret_whu_swinb --checkpoint_name hyret_whu_swinb_ckpt.pt

# Swin-Base + Cityscapes Semantic Segmentation
python eval_cd.py --data_name WHU --project_name hyret_whu_swinb_citysem --checkpoint_name hyret_whu_swinb_citysem_ckpt.pt

# Mamba1k
python eval_cd.py --data_name WHU --project_name hyret_whu_mamba1k --checkpoint_name hyret_whu_mamba1k_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation
python eval_cd.py --data_name WHU --project_name hyret_whu_resnet50_flip30 --checkpoint_name hyret_whu_resnet50_flip30_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation + 30% Crop
python eval_cd.py --data_name WHU --project_name hyret_whu_resnet50_flip30_crop --checkpoint_name hyret_whu_resnet50_flip30_crop_ckpt.pt

# ResNet-50 + 30% Flip + 30% Rotation + 30% Crop + CE+Dice Loss
python eval_cd.py --data_name WHU --project_name hyret_whu_resnet50_flip30_crop_ce_dice --checkpoint_name hyret_whu_resnet50_flip30_crop_ce_dice_ckpt.pt
```

---

#### [Mustansar Fiaz](https://sites.google.com/view/mustansarfiaz/home), [Mubashir Noman](https://scholar.google.com/citations?user=S6_CVskAAAAJ&hl=en),  [Hiyam Debary](https://www.linkedin.com/in/hiyam-debary/), [Kamran Ali](https://scholar.google.com/citations?user=JuQ_vNIAAAAJ&hl=en), [Hisham Cholakkal](https://hishamcholakkal.com/)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2506.12836)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FF9900)](https://huggingface.co/mustansarfiaz/HyRet)


---
### 🏆 Highlights
-----------------
- **HyRet-Change:** We propose a Siamese-based framework, which can seamlessly integrate the merits of convolution and retention mechanisms at multi-scale features to preserve critical information and enhance adaptability in complex scenes change detection (CD).  Specifically, we propose a hybrid plug-and-play feature difference module (FDM) to explore rich feature information utilizing both self-attention and convolution operations in a parallel way. This unique integration, at multi-scale features, leverages the
advantages of both local features and long-range contextual information. We introduce a retention mechanism in our novel FDM to mitigate the limitations of standard self-attention.
- **Local-Global (LG)-Interaction Module:** We introduce an adaptive interaction between local and global representations to exploit the intricate relationship contextually to strengthen the model’s ability to perceive meaningful changes while reducing the effect of pseudo-changes.
- **Experiments:** Our extensive experimental study over three challenging CD datasets demonstrates the merits of our approach while achieving state-of-the-art performance.

---
### 👁️💬 Proposed Framework
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/fig1.png">

---
###  📊 Quantitative Comparison
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/table1.png">

---
### :speech_balloon: Qualitative Comparison from the LEVIR-CD (first row) and WHU-CD (second row) datasets
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/qualitative.png">

---

### Requirements
```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2
```

---
### :speech_balloon: Dataset Preparation

### :point_right: Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

---
## Citation

```
@inproceedings{fiaz2025hyret,
  title={HyRet-Change: A hybrid retentive network for remote sensing change detection},
  author={Fiaz, Mustansar and Noman, Mubashir and Debary, Hiyam and Ali, Kamran and Cholakkal, Hisham},
  booktitle={IGARSS 2025-2025 IEEE International Geoscience and Remote Sensing Symposium},
  year={2025},
  publisher={IEEE}
}
@inproceedings{noman2024changebind,
  title={Changebind: A hybrid change encoder for remote sensing change detection},
  author={Noman, Mubahsir and Fiaz, Mustansar and Cholakkal, Hisham},
  booktitle={IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium},
  pages={8417--8422},
  year={2024},
  organization={IEEE}
}
@article{noman2024elgc,
  title={ELGC-Net: Efficient local--global context aggregation for remote sensing change detection},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham and Khan, Salman and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--11},
  year={2024},
  publisher={IEEE}
}
@article{noman2024remote,
  title={Remote sensing change detection with transformers trained from scratch},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham and Narayan, Sanath and Anwer, Rao Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--14},
  year={2024},
  publisher={IEEE}
}
```

### Contact

If you have any question, please feel free to contact the authors. Mustansar Fiaz: [mustansar.fiaz@ibm.com](mailto:mustansar.fiaz@ibm.com) or Mubashir Noman: [mubashir.noman@mbzuai.ac.ae](mailto:mubashir.noman@mbzuai.ac.ae).

## References
Our code is based on [Changebind](https://github.com/techmn/changebind) repository. 
We thank them for releasing their baseline code.

