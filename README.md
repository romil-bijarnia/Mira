# Skin Cancer Detector – Research-grade Example

This repository now contains a **deep-learning** pipeline for skin cancer
classification using *transfer learning*.  It targets the
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
dataset but can operate on any image folder organised by class.

The implementation uses **PyTorch** with a pretrained ResNet-18 backbone and
offers command-line options for training, evaluation and saving model weights.

> ⚠️ **Disclaimer** – This software is **not** a medical device and must not be
> used for clinical decisions.  It is intended for research and educational
> purposes only.

## Training on the HAM10000 dataset

1. Download and extract the dataset (images and `HAM10000_metadata.csv`).
2. Run the training script:

   ```bash
   python main.py \
       --metadata /path/to/HAM10000_metadata.csv \
       --images-dir /path/to/all_images/ \
       --epochs 10 --batch-size 32
   ```

   The best-performing model on the validation set will be saved to
   `skin_cancer_resnet18.pth` by default.

## File overview

* **`main.py`** – PyTorch training script with dataset loader and
  transfer-learning model.
* **`README.md`** – you are here.

## Requirements

* Python ≥ 3.9
* [PyTorch](https://pytorch.org/)
* [torchvision](https://pytorch.org/vision/stable/index.html)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)

These dependencies can be installed with:

```bash
pip install torch torchvision pandas scikit-learn
```

---

Happy researching! 🧪

