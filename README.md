<p align="center">
  <h1 align="center">Illumination-Invariant Representation Learning with Multiple Priors for Low-Light Object Detection
</h1>
  <p align="center">
    Qian Zhao
    ·
    Gang Li
    ·
    Tao Pang
    ·
    Mingke Gao
  </p>
  

## 📢 Abstract

Detecting objects under low-light conditions remains a formidable challenge due to reduced visibility and severe information degradation. Recent approaches attempt to learn illumination-invariant representations to enhance detection performance in dark environments. Nonetheless, these methods often rely on handcrafted priors derived from idealized imaging assumptions, which fundamentally limit their robustness and generalization under real-world lighting conditions. To overcome the modeling limitations of physics-based priors under non-ideal illumination, we propose three complementary representations learning  strategies, including frequency-domain decomposition, geometric prior, and language guidance, to jointly learn more robust illumination-invariant features. First, we introduce a Frequency–Spatial Reconstruction Module (FSRM) that discards illumination-sensitive amplitude information and retains only the phase spectrum in the Fourier domain to extract structurally stable cues such as edges and contours. Second, we incorporate geometric priors derived from a depth estimation backbone to capture spatial layouts invariant to lighting variations, and design a Geometry-Aware Feature Enhancement (GAFE) module to enable depth-guide representation learning. Third, we leverage the vision-language alignment capability of CLIP by introducing textual prompts as semantic supervision, guiding the model to learn representations that are steered away from undesirable characteristics associated with darkness or degradation. These three strategies, respectively grounded in structural, spatial, and semantic perspectives, collaboratively mitigate illumination-induced degradations and significantly improve the robustness and generalization of the learned features. Extensive experiments demonstrate that our method achieves state-of-the-art performance, attaining 83.2\% mAP50 on ExDark and 70.3\% mAP50 on DARK FACE. The code is publicly available at https://github.com/dlgxwcvhehks/MIPNet.




## 📜 Network Architecture
<p align="center">
<img src=https://github.com/dlgxwcvhehks/MIPNet/blob/main/Architecture.png width="750px" height=400px">
</p>



## :rocket: Installation

The model is implemented using PyTorch 2.1.0 and has been tested with Python 3.10, CUDA 11.8, and Detectron2 0.6. Please refer to the instructions below for the installation details.

```
conda create -n your_env python=3.10
conda activate your_env

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install -e detectron2

```

## :notebook_with_decorative_cover: Dataset Preparation

### ExDark Dataset: 
Detectron2 requires annotations in JSON format.You can download the original ExDark dataset provided by MAET (in VOC format) from [Google Drive](https://github.com/cuiziteng/ICCV_MAET?tab=readme-ov-file) and convert the annotations to JSON, or directly use the JSON annotations we provide for convenience from the [link](https://drive.google.com/file/d/1JqgRdKrg_PchxFmkQ90gW9iPnCSB_kgx/view?usp=drive_link).

The dataset structure should be like:
```
├── exdark
│   ├── test
│   │   ├── 2015_00004.jpg
│   │   ├── 2015_00014.jpg
│   │   ├── xx.jpg
│   ├── train
│   │   ├── 2015_00001.png
│   │   ├── 2015_00020.jpg
│   │   ├── xx.jpg
├── exdark_test.json
├── exdark_train.json
```

### DARK FACE Dataset:
Similarly, you can download the original DarkFace dataset provided by MAET (in its official annotation format) from [GoogleDrive](https://github.com/cuiziteng/ICCV_MAET?tab=readme-ov-file) and convert the annotations to JSON accordingly, or directly use the JSON annotations we provide for convenience from the [link](https://drive.google.com/file/d/1xp2kzLiZd8faT84_FCv8lLCJrNsCNcXI/view?usp=drive_link).

The dataset structure should be like:
```
├── darkface
│   ├── test
│   │   ├── 9.png
│   │   ├── 18.png
│   │   ├── xx.png
│   ├── train
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── xx.png
├── darkface_test.json
├── darkface_train.json
```

## :notebook: Model Preparation
### Download Pretrained Backbones

Before running evaluation, please download the required pretrained models:


(1) QueryRCNN Backbone: Please download [queryrcnn_r101_cascade_300pro_3x_f7617a4d.pth](https://drive.google.com/file/d/1Gd8XYFiS0cxVKo-UPAZ_3G4AaescmH2X/view?usp=drive_link) and place it under MIPNet/MIPNet-main/checkpoint/.

(2) Depth Estimation Model: Please download [depth_anything_v2_vits.pth](https://drive.google.com/file/d/1VyYBHX4jd1zWnO9-7QDzTyzfcwzwbG1d/view?usp=drive_link) and place it under MIPNet/MIPNet-main/queryrcnn/Illumination_Invariant/depth_anything_v2/.

(3) CLIP Model: Please download [https://drive.google.com/file/d/1QUBfWar4xwKzPMDd7Cm-3_MomO7ZIhCU/view?usp=drive_link) and place it under MIPNet/MIPNet-main/queryrcnn/CLIP/.

### Download Our Trained Models
Please download the trained checkpoints and place it under MIPNet/MIPNet-main/checkpoint/.
| Datasets  | Checkpoints           |
|------------|-----------------------|
| ExDark     | [model_exdark.pth](https://drive.google.com/file/d/1zT-Kj3nRDPp0b8SItmhB0fcZoWVdUt5m/view?usp=drive_link)      |
| DarkFace   | [https://drive.google.com/file/d/1l0oABe2Obkt841sitN6DBIJyYjKl4OGK/view?usp=drive_link)    |

## 📡 Evaluation

### Evaluate on ExDark
```
python test_exdark.py \
    --config-file configs/exdark_config_test.yaml \
    --eval-only \
    --num-gpus 1 \
    MODEL.WEIGHTS checkpoint/model_exdark.pth
```
### Evaluate on DarkFace

```
python test_darkface.py \
    --config-file configs/darkface_config_test.yaml \
    --eval-only \
    --num-gpus 1 \
    MODEL.WEIGHTS checkpoint/model_darkface.pth
```

## 📑 Citation

If you find this work useful, please cite


``` citation
@article{MIPNet,
  title={Illumination-Invariant Representation Learning with Multiple Priors for Low-Light Object Detection},
  author={Q. Zhao, G. Li, T. Pang, and M. Gao},
  journal={},
  year={2026}
}
```





