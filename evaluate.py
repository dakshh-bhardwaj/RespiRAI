import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
DATA_DIR = 'd:/pneumonia/chest_xray'
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = 'd:/pneumonia/pneumonia_model.keras'

print("Loading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
  TEST_DIR,
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE,
  shuffle=False)

class_names = test_ds.class_names
print(f"Class names: {class_names}")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Evaluating model...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy:.4f}")

print("Generating predictions for detailed metrics...")
y_pred_logits = model.predict(test_ds)
y_pred = np.argmax(y_pred_logits, axis=1)

y_true = np.concatenate([y for x, y in test_ds], axis=0)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('d:/pneumonia/confusion_matrix.png')
print("Confusion matrix saved to d:/pneumonia/confusion_matrix.png")
