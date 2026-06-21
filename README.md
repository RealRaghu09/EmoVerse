# Emoverse – Speech Emotion Recognition using Deep Learning

## Overview

EmoVerse is a Deep Learning-based Speech Emotion Recognition (SER) system that classifies human emotions from speech audio signals.

The model analyzes audio recordings and predicts emotions such as:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fear
* Disgust
* Surprise

This project was developed using audio signal processing techniques and Convolutional Neural Networks (CNNs) to understand emotional patterns in speech.

---

# Problem Statement

Humans naturally express emotions through speech, but machines cannot easily understand emotional context.

The objective of this project is to build an intelligent system capable of recognizing emotions from speech recordings using Deep Learning.

Applications include:

* Virtual Assistants
* Human-Computer Interaction
* Mental Health Monitoring
* Customer Support Analysis
* Voice Analytics Systems

---

# Dataset

## RAVDESS Dataset

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

### Dataset Features

* 24 professional actors
* High-quality WAV audio files
* Balanced emotional speech recordings
* Multiple emotional classes

### Emotion Labels

| Label | Emotion  |
| ----- | -------- |
| 0     | Neutral  |
| 1     | Calm     |
| 2     | Happy    |
| 3     | Sad      |
| 4     | Angry    |
| 5     | Fear     |
| 6     | Disgust  |
| 7     | Surprise |

---

# Project Workflow

## 1. Data Loading

* Loaded audio files from the dataset directory
* Extracted emotion labels from filenames
* Stored paths and labels for preprocessing

Example filename:

```bash
03-01-05-01-02-02-12.wav
```

Here:

```bash
05 → Angry Emotion
```

---

## 2. Audio Visualization

Audio visualization was performed to better understand emotional speech patterns.

### Techniques Used

#### Waveplots

Used to analyze:

* Amplitude variations
* Speech intensity
* Temporal characteristics

#### Spectrograms

Used to visualize:

* Frequency distribution
* Energy variations
* Emotional tone differences

---

## 3. Data Augmentation

To improve model generalization and reduce overfitting, multiple augmentation techniques were applied.

### Augmentation Techniques

#### Noise Injection

Adds random noise to audio samples.

Purpose:

* Improves robustness
* Handles noisy environments

---

#### Time Stretching

Changes audio speed without affecting pitch.

Purpose:

* Simulates different speaking rates

---

#### Audio Shifting

Shifts audio forward or backward in time.

Purpose:

* Improves positional invariance

---

#### Pitch Shifting

Modifies pitch frequencies.

Purpose:

* Simulates speaker variations

---

# Feature Extraction

Raw audio signals are high-dimensional and noisy. Feature extraction helps capture meaningful emotional information.

## Features Used

### MFCC (Mel-Frequency Cepstral Coefficients)

Captures:

* Human auditory characteristics
* Vocal tract information

---

### Mel Spectrogram

Captures:

* Frequency-energy distribution

---

### Chroma Features

Captures:

* Tonal information
* Pitch class distribution

---

### Zero Crossing Rate (ZCR)

Captures:

* Signal activity
* Emotional intensity

---

### Root Mean Square (RMS)

Captures:

* Loudness
* Signal energy

---

# Model Architecture

A 1D Convolutional Neural Network (CNN) was used for emotion classification.

## Layers Used

### Convolutional Layers

* Learn local audio patterns
* Extract emotional representations

### MaxPooling Layers

* Reduce dimensionality
* Preserve important features

### Dropout Layers

* Prevent overfitting
* Improve generalization

### Dense Layers

* Perform final emotion classification

---

# Training Details

| Parameter         | Value            |
| ----------------- | ---------------- |
| Framework         | PyTorch          |
| Model Type        | 1D CNN           |
| Optimizer         | Adam             |
| Loss Function     | CrossEntropyLoss |
| Evaluation Metric | Accuracy         |
| Dataset Split     | Train/Test       |

---

# Model Performance

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 82.87% |

---

# Observations

* Training loss decreased consistently
* Validation accuracy stabilized around 69%
* Emotions like Happy and Surprise showed overlap
* Data augmentation improved generalization

---

# Challenges Faced

## Limited Dataset Size

Speech emotion datasets are relatively small.

### Solution

* Applied data augmentation
* Used dropout regularization

---

## Similar Emotional Patterns

Examples:

* Happy vs Surprise
* Calm vs Neutral

### Solution

* Used multiple complementary audio features

---

## Overfitting

### Solution

* Dropout layers
* Data augmentation
* Proper train-validation split

---

---

# Tech Stack

| Category             | Tools               |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Deep Learning        | PyTorch             |
| Audio Processing     | Librosa             |
| Data Handling        | NumPy, Pandas       |
| Visualization        | Matplotlib, Seaborn |

---

# Conclusion

EmoWave demonstrates how Deep Learning and Audio Signal Processing can be combined to recognize human emotions from speech audio.

The project provided hands-on experience in building end-to-end AI systems involving:

* preprocessing
* feature extraction
* model training
* evaluation
* emotion classification
