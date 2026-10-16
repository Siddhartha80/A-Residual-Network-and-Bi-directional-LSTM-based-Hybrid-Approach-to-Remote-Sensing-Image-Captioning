<div align="center">

# 🛰️ A Residual Network and Bi-directional LSTM based Hybrid Approach to Remote Sensing Image Captioning

### ResNet50 + Bi-LSTM encoder–decoder that describes satellite imagery in natural language

[![Paper](https://img.shields.io/badge/Published-Procedia%20Computer%20Science%202025-005EB8?logo=elsevier&logoColor=white)](https://www.sciencedirect.com/science/article/pii/S1877050925013006)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RSICD-informational)
![Task](https://img.shields.io/badge/Task-Image%20Captioning-blueviolet)

**📄 Published in *Procedia Computer Science* (Elsevier, 2025):**
[**Read the paper on ScienceDirect →**](https://www.sciencedirect.com/science/article/pii/S1877050925013006)

</div>

---

## 📖 Overview

Remote sensing imagery is essential for monitoring the Earth's surface, supporting land-use analysis, agriculture, environmental monitoring, and disaster relief. Turning those images into **natural-language descriptions** makes them far easier to interpret and act on.

This work proposes a **hybrid encoder–decoder model** for remote sensing image captioning:

- **Encoder:** a pretrained **ResNet50** extracts rich spatial and spectral features from each image.
- **Decoder:** a **Bi-directional LSTM (Bi-LSTM)** reads the caption sequence in both directions and, fused with the image features, generates the description word by word.

The approach is benchmarked against other CNN backbones (VGG16, DenseNet201, EfficientNetB0, InceptionNet) combined with LSTM decoders, using **BLEU, METEOR and ROUGE** metrics.

---

## 🏆 Key Results

The proposed **ResNet50 + Bi-LSTM** model achieved the best scores among the compared models:

| Metric | Score |
|--------|------:|
| **BLEU-1** | **75.45%** |
| **BLEU-2** | **65.98%** |
| **BLEU-3** | **56.54%** |
| **BLEU-4** | **44.64%** |
| **ROUGE** | **42.65%** |
| **METEOR** | **40.69%** |

These results are as reported in the [published paper](https://www.sciencedirect.com/science/article/pii/S1877050925013006), which also shows the model generalizing better than the baseline architectures.

---

## 🧠 Architecture

<div align="center">
<img src="Architecture%20Diagrams/DLSIP.drawio.png" alt="ResNet50 + Bi-LSTM captioning architecture" width="320"/>
</div>

| Component | Details |
|-----------|---------|
| **Image branch** | ResNet50 (ImageNet weights), penultimate-layer 2048-d features → Dense(256) → Reshape(1, 256) |
| **Text branch** | Tokenized caption → Embedding(256) → Bidirectional LSTM |
| **Fusion** | Image features concatenated with the caption sequence, then passed through the Bi-LSTM |
| **Head** | Dense layers (512 → 128) with Dropout → Softmax over the vocabulary |
| **Training** | Categorical cross-entropy, Adam, up to 50 epochs, early stopping on validation loss |

### Pipeline

<div align="center">
<img src="Architecture%20Diagrams/system.jpg" alt="Project workflow" width="200"/>
</div>

`Data collection & preprocessing → CNN feature extraction → Caption generation model → Training → Evaluation (BLEU / METEOR / ROUGE)`

---

## 📊 Dataset

**[RSICD](https://arxiv.org/abs/1712.07835)** (Remote Sensing Image Captioning Dataset): **10,921** aerial and satellite images, each paired with **five human-written captions** (airports, beaches, farmland, residential areas, and more).

- Official split in `dataset_rsicd.json`: **8,734 train / 1,094 validation / 1,093 test** images
- Captions are lower-cased, cleaned, and wrapped with `startseq` / `endseq` tokens
- `Modified_Book2.xlsx` holds the cleaned image–caption table used by the notebooks
- The RSICD images are **not** included; `DataSet Loading.ipynb` shows how to fetch the dataset from Hugging Face (`arampacha/rsicd`)

---

## 🗂️ Repository Structure

```
.
├── Architecture Diagrams/        # Model architecture, workflow and training-curve figures
├── DataSet Loading.ipynb         # Download RSICD from Hugging Face
├── ResNet (BiLstm).ipynb         # ⭐ Proposed model: ResNet50 + Bi-LSTM, training and BLEU/METEOR/ROUGE evaluation
├── Fi.ipynb                      # DenseNet201 feature + LSTM captioning baseline with a custom data generator
├── DenseNet201.ipynb             # DenseNet201 comparison
├── EfficientNetB0.ipynb          # EfficientNetB0 comparison
├── VGG16.ipynb                   # VGG16 comparison
├── inception_fin.ipynb           # InceptionNet comparison
├── res.ipynb · vgg.ipynb · dense201.ipynb   # Additional experiment notebooks
├── dataset_rsicd.json            # RSICD captions and train/val/test split
├── Modified_Book2.xlsx           # Cleaned image–caption table
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Getting Started

### Installation

```bash
git clone https://github.com/Siddhartha80/A-Residual-Network-and-Bi-directional-LSTM-based-Hybrid-Approach-to-Remote-Sensing-Image-Captioning.git
cd A-Residual-Network-and-Bi-directional-LSTM-based-Hybrid-Approach-to-Remote-Sensing-Image-Captioning

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The NLTK data downloads are needed for the BLEU and METEOR evaluation cells.

### Run

1. Get the **RSICD images** (see `DataSet Loading.ipynb` for loading the dataset from Hugging Face).
2. Open **`ResNet (BiLstm).ipynb`** and update the image and caption paths at the top. They are currently Windows paths such as `C:\Users\...\RSICD_images\`.
3. Launch Jupyter and run the cells in order:

```bash
jupyter notebook
```

The notebook extracts ResNet50 features, trains the Bi-LSTM decoder, and reports BLEU-1 to BLEU-4, ROUGE and METEOR. It also includes single-image caption generation demos.

> The feature-extraction cell processes a limited number of images at a time to keep memory use manageable. Increase this limit if your machine has more RAM.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow / Keras (ResNet50, VGG16, DenseNet201, EfficientNetB0, InceptionNet, LSTM / Bi-LSTM)
- **NLP & evaluation:** NLTK (BLEU, METEOR), `rouge-score`, Keras `Tokenizer`
- **Data & vision:** NumPy, Pandas, OpenCV, Hugging Face `datasets`
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## 📚 Citation

If you use this work, please cite the published paper:

> *A Residual Network and Bi-directional LSTM based Hybrid Approach to Remote Sensing Image Captioning.* **Procedia Computer Science**, Elsevier, 2025.
> https://www.sciencedirect.com/science/article/pii/S1877050925013006

**Dataset:** Lu, X., Wang, B., Zheng, X., & Li, X. (2018). *Exploring Models and Data for Remote Sensing Image Caption Generation.* IEEE Transactions on Geoscience and Remote Sensing. [arXiv:1712.07835](https://arxiv.org/abs/1712.07835)

---

## ⚠️ Disclaimer

This repository is for research and educational purposes. Generated captions are model predictions and may contain errors, so they should be reviewed before being used in any operational or decision-making context.

---

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>
