# EmoVerse
Helps understanding the emotion by the voice of the person
Data sets - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

This project builds a deep learning model that can recognize emotions from speech. Using the RAVDESS dataset, the model is trained to identify eight different emotions: neutral, calm, happy, sad, angry, fearful, disgusted, and surprised.
The workflow starts with cleaning and preparing the audio data, which includes reducing noise, trimming silence, and normalizing the signals. Next, we extract meaningful features from the audio, such as MFCCs, chroma features, and spectrograms, that help the model understand the unique patterns of each emotion. To make the system more robust, we also apply data augmentation—for example, slightly changing pitch, stretching time, or adding background noise to simulate real-world conditions.
For the model itself, we use a hybrid deep learning approach, combining CNN layers to capture frequency patterns in the audio and RNN layers (like LSTMs or GRUs) to understand time-based changes in speech. The model is trained with modern optimization techniques and fine-tuned to avoid overfitting.
Finally, the system is tested using multiple evaluation metrics (accuracy, precision, recall, and F1-score), giving a clear picture of how well it can detect emotions. Overall, this project shows how AI can be used to make machines more emotionally aware, opening doors for applications in virtual assistants, customer service, mental health tools, and interactive voice systems.

##1. Data Augmentation
To improve model robustness and generalization, the following augmentation techniques are applied to the audio signals:

Noise Injection – Adds random background noise to simulate real-world conditions.

Time-Stretching – Alters the speed of the audio without changing pitch.

Shifting – Slightly shifts the audio waveform along the time axis.

Pitch Adjustment – Modifies the pitch to introduce vocal variations.

##2. Feature Extraction
To capture meaningful acoustic properties while minimizing noise and complexity, the following features are extracted:

Zero-Crossing Rate (ZCR): Measures the rate of sign changes in the signal, useful for detecting speech energy.

Chroma Features: Represents the intensity of different pitch classes, capturing harmonic and tonal content.

Mel-Frequency Cepstral Coefficients (MFCCs): Encodes timbral aspects of speech, widely used in audio analysis.

Mel-Spectrogram: Provides a time-frequency representation aligned with human auditory perception.

Root Mean Square (RMS): Estimates the energy of the signal, highlighting loudness variations.

##3. Model Training
The classification model is built using a 1D Convolutional Neural Network (CNN) optimized for sequential audio features. Its architecture includes:

Convolutional Layers: Extract spatial and frequency-related features from the input.

MaxPooling Layers: Downsample feature maps to reduce dimensionality and computation.

Dropout Layers: Regularize the model by preventing overfitting.

Dense (Fully Connected) Layers: Map the extracted features to the final output classes.

The final dense layer classifies the audio signals into eight emotion categories: neutral, calm, happy, sad, angry, fear, disgust, and surprise.
