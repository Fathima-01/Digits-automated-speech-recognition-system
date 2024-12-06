# Digits-automated-speech-recognition-system
Convert the speech(digits) into text 

This project focuses on building a machine learning model for recognizing spoken digits (0-9) using the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset). The implementation includes preprocessing audio data, extracting MFCC features, and training a neural network for classification.


## **Overview**

This project uses MFCC-based feature extraction and a neural network model to classify audio recordings of spoken digits. The goal is to convert speech signals into corresponding text digits with high accuracy.

---

## **Features**

- Extracts MFCC features from audio files.  
- Implements a neural network using TensorFlow/Keras.  
- Supports inference for custom `.wav` files.  
- Includes model training and evaluation scripts.

---

## **Dataset**

- Dataset: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)  
- Number of recordings: 1,500  
- Speakers: 6  
- Digits: 0-9  
- Sampling rate: 8 kHz  

To use the dataset:  
1. Download the dataset from the provided link.  
2. Place it in the directory: `free-spoken-digit-dataset/recordings`.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **1. Training the Model**
Run the following script to train the model:
```bash
python train.py
```
This script:
- Extracts MFCC features.
- Splits the data into training and test sets.
- Trains the model and saves it to the `models/` directory.

### **2. Testing the Model**
Evaluate the model using:
```bash
python evaluate.py
```
This will print the test accuracy and display sample predictions.

### **3. Inference**
To make predictions on a custom audio file:
```bash
python predict.py --file <path-to-audio-file>
```
Replace `<path-to-audio-file>` with the path of your `.wav` file.

---

## **Pre-trained Model**

Download the pre-trained model from [here](#).  
Place the model in the `models/` directory for quick inference.

---

## **Results**

- **Test Accuracy**: 80.33%
- Model performed well on the test set, with occasional misclassifications on similar-sounding digits (e.g., 6 and 8).

---

## **Future Work**

- Add support for noisy and accented datasets.  
- Extend the vocabulary to include words or alphanumeric sequences.  
- Explore advanced architectures like CNNs and Transformers for better accuracy.

---

Feel free to use this README in your GitHub repository. Let me know if you'd like to customize it further or include additional sections!
