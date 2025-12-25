# Nepali–English Code-Mixed Automatic Speech Recognition (ASR)

This repository contains an **Automatic Speech Recognition (ASR)** system designed for **Nepali–English code-mixed speech**, built using a **custom fine-tuned Wav2Vec2 model**.

The project is part of a larger goal:

> **Real-Time Speech Recognition and Romanized Nepali-to-Devanagari Bilingual Subtitle Generation for Informative Content**

---

## Motivation

In real-world Nepali media (YouTube videos, interviews, podcasts, educational content), speakers frequently **mix Nepali and English within the same sentence**.  
Most existing ASR systems struggle with:
- Code-switching
- Romanized Nepali
- Low-resource Nepali speech data

This project aims to address these challenges by building a **custom ASR pipeline** tailored for **Nepali–English code-mixed speech**.

---

## Project Highlights

- Custom ASR pipeline using **HuggingFace Transformers**
- Fine-tuning **facebook/wav2vec2-large-xlsr-53** (317M parameters)
- Character-level vocabulary for code-mixed text
- Data augmentation for low-resource speech
- Evaluation using **WER** and **CER**
- End-to-end training, evaluation, and testing pipeline

---

## 📁 Repository Structure
nepali-english-asr/

│

├── data/ # Audio files and metadata CSV

├── scripts/ # Training, preprocessing, and setup scripts

├── utils/ # Data collator, metrics, helper functions

├── processor/ # Saved tokenizer & feature extractor

├── final_model/ # Trained model checkpoints

├── test_samples/ # Sample audio files for inference testing

├── requirement.txt # Python dependencies

└── README.md


---

##  Model Architecture

- **Base Model:** `facebook/wav2vec2-large-xlsr-53`
- **Training Objective:** CTC (Connectionist Temporal Classification)
- **Tokenizer:** Character-level tokenizer built from transcripts
- **Languages Supported:**  
  - Romanized Nepali  
  - English  
  - Code-mixed Nepali–English speech

---

##  Dataset

- **Original dataset size:** 500 audio samples
- **After augmentation:** ~3000 samples
- **Augmentation techniques used:**
  - Speed perturbation
  - Pitch shifting
  - Random background noise

>  Despite augmentation, dataset size remains a limitation for achieving low WER.

---

##  Evaluation Metrics

The model is evaluated using:

- **Word Error Rate (WER)**  
  Measures word-level transcription accuracy.

- **Character Error Rate (CER)**  
  Important for:
  - Character-level tokenization
  - Low-resource languages
  - Code-mixed speech

Both metrics are computed during validation and testing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Siddhartha-Thapa/nepali-english-asr.git
cd nepali-english-asr
```
Install dependencies:

```bash
pip install -r requirement.txt
```
## Training Pipeline
**1️⃣ Dataset Preparation**

Audio files organized

Metadata CSV created

Train / Validation / Test split

HuggingFace Dataset created

**2️⃣ Processor Setup**

Character-level vocabulary generation

Wav2Vec2 Processor initialization

**3️⃣ Model Training**

Pretrained Wav2Vec2 loaded

Dynamic padding using CTC Data Collator

Training executed using HuggingFace Trainer

**4️⃣ Evaluation**

CER and WER computed

Model tested on unseen audio samples

**▶️ Usage**
Train the model
```bash
python scripts/train.py
```
Evaluate the model
```bash
python scripts/evaluate.py
```

Test on sample audio
```bash
python scripts/inference.py
```

Script names may vary — refer to the scripts/ directory for exact filenames.

Known Limitations

Dataset size is insufficient for production-level ASR

High WER observed despite reasonable CER

Wav2Vec2 struggles with heavy code-mixing in low-resource settings

Future Work

Increase dataset size with real-world Nepali–English speech

Experiment with OpenAI Whisper models

Improve Romanized Nepali → Devanagari conversion

Deploy real-time ASR + subtitle generation pipeline

## Contributions

Contributions are highly welcome 🎉

You can contribute by:

Adding more speech data

Improving preprocessing or training strategies

Experimenting with alternative ASR models (e.g., Whisper)

Improving evaluation or inference scripts

Steps:

Fork the repository

Create a new branch

Commit your changes

Open a Pull Request




## 👤 Author

Siddhartha Thapa
Computer Engineering Student

If you find this project useful or want to collaborate, feel free to connect on LinkedIn or contribute to the repository .
