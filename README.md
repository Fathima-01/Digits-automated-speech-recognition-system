# Digits-automated-speech-recognition-system
Convert the speech(digits) into text 

This project demonstrates how to build a machine learning model to recognize spoken digits (0-9) using the Free Spoken Digit Dataset (FSDD). The model is trained using the TensorFlow/Keras framework and uses MFCC-based feature extraction for digit recognition. The project is designed to be run on **Google Colab**, enabling quick execution and testing.

## **Introduction**

Speech recognition is a core AI application that allows machines to understand and process human speech. This project focuses on recognizing spoken digits (0-9) and converting them into corresponding text using the Free Spoken Digit Dataset (FSDD). The model is built and trained on Google Colab, which provides an accessible environment with GPU support for faster processing.

The project is structured to:
- Preprocess audio data.
- Extract MFCC features.
- Train a neural network to classify spoken digits.
- Make predictions on new audio files.


## **Features**

- **Feature Extraction**: Uses MFCC features for speech recognition.
- **Neural Network Model**: Built using TensorFlow/Keras for digit classification.
- **Google Colab Compatibility**: Easily runnable on Google Colab for immediate access to GPU support.
- **Inference**: Make predictions on new `.wav` files.


## **Dataset**

The project uses the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset), which contains clean audio recordings of digits 0 through 9 spoken by multiple speakers.

- **Number of Recordings**: 1,500  
- **Number of Speakers**: 6 (George, Jackson, Nicolas, Theo, Yweweler, Lucas)  
- **Digits**: 0-9  
- **Sampling Rate**: 8 kHz  
- **File Format**: `.wav`  
- **Duration**: ~0.5 seconds per recording  

To use the dataset:
1. Download it from [here](https://github.com/Jakobovski/free-spoken-digit-dataset).  
2. Upload it to Google Colab or store it in a folder named `free-spoken-digit-dataset/recordings`.


## **Installation**

### **Running on Google Colab**

To run this project on Google Colab, follow these steps:

1. **Clone the repository**:
   - Open a new Colab notebook and run the following command:
     ```python
     !git clone <repository-link>
     %cd <repository-folder>
     ```

2. **Install the required dependencies**:
   - Install the necessary Python packages by running:
     ```python
     !pip install -r requirements.txt
     ```

3. **Upload the Dataset**:
   - Either upload the dataset to Google Colab manually or use the following code to download it directly from the GitHub repository:
     ```python
     !git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
     ```


## **Project Structure**

The repository is organized as follows:

```
├── free-spoken-digit-dataset/      # Dataset folder
│   └── recordings/                # Audio files
├── models/                        # Directory to store trained models
├── train.py                       # Training script
├── evaluate.py                    # Model evaluation script
├── predict.py                     # Inference script
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
└── utils.py                       # Utility functions (feature extraction, etc.)
```


## **Usage**

### **Running on Google Colab**

1. **Training the Model**:
   After uploading the dataset and installing dependencies, run the following to train the model:
   ```python
   !python train.py
   ```

   This will:
   - Preprocess the data and extract MFCC features.
   - Train the neural network model.
   - Save the model in the `models/` directory.

2. **Evaluating the Model**:
   To evaluate the trained model, use:
   ```python
   !python evaluate.py
   ```

   This will print the accuracy of the model on the test set.

3. **Making Predictions**:
   To make predictions on new `.wav` files, use the following:
   ```python
   !python predict.py --file <path-to-audio-file>
   ```

   Replace `<path-to-audio-file>` with the path to the `.wav` file you want to predict.


## **Pre-trained Model**

A pre-trained model is available for quick inference.  
You can download the pre-trained model from [here](#).  
Once downloaded, upload the model to Google Colab and place it in the `models/` directory.


## **Results**

The model achieved an **accuracy of 80.33%** on the test set. Below is the training and validation performance over 10 epochs:

| **Epoch** | **Training Accuracy (%)** | **Validation Accuracy (%)** | **Validation Loss** |  
|-----------|----------------------------|------------------------------|----------------------|  
| 1         | 19.10                      | 38.17                        | 2.72                 |  
| 5         | 70.99                      | 73.83                        | 0.72                 |  
| 10        | 78.88                      | 80.33                        | 0.53                 |  

The model performs well on clean speech but can be further enhanced by introducing noise robustness and additional training data.


## **Future Enhancements**

1. **Noise Robustness**:  
   Train the model on noisy datasets or introduce data augmentation techniques.

2. **Advanced Architectures**:  
   Explore convolutional neural networks (CNNs) or Transformer models for improved accuracy.

3. **Expanded Vocabulary**:  
   Extend the model to recognize words or larger alphanumeric sequences.

4. **Real-Time Recognition**:  
   Integrate the system into a real-time speech-to-text application for live recognition.


This project is designed to be easily run and tested on Google Colab. Feel free to modify the code and expand the model for more advanced applications!
