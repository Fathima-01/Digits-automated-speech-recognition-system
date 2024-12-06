import numpy as np
from tensorflow.keras.models import load_model
from train import load_data

# Load the test data
_, X_test, _, y_test = load_data()

# Load the trained model
model = load_model('models/digit_recognition_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
