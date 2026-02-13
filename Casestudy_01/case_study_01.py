import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Dataset Handling 

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Original Shape of x_train:", x_train.shape)
print("Original Shape of x_test :", x_test.shape)

# Reshape (28x28 -> 784) and Normalize (0-255 -> 0-1)
x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0

print("Reshaped Shape of x_train:", x_train.shape)
print("Reshaped Shape of x_test :", x_test.shape)

# MLP Model Architecture 

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),   # Hidden Layer
    Dense(64, activation='relu'),                        # Hidden Layer
    Dense(10, activation='softmax')                      # Output Layer
])

print("\nModel Summary:")
model.summary()

# 3. Model Compilation 

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Training the Model 

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# 5. Model Evaluation 

test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)


# 6. Visualization 

# Plot 1: Training vs Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Training vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Loss vs Epochs
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# 7. Observations and Result Explanation (Mandatory)

print("\n---------------- OBSERVATIONS ----------------")
print("1. The model achieved test accuracy of:", round(test_acc*100, 2), "%")
print("2. Training accuracy increases with epochs, showing learning progress.")
print("3. Validation accuracy also increases and remains close to training accuracy.")
print("4. If validation loss starts increasing while training loss decreases, overfitting occurs.")
print("5. Here, the model performs well and shows good generalization.")