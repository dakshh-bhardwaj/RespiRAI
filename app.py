import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'd:/pneumonia/pneumonia_model.keras'
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def predict_image(image_path):
    img = load_img(image_path, target_size=(180, 180))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predicted_class, confidence = predict_image(filepath)
            
            return render_template('index.html', 
                                   image_path=filepath, 
                                   prediction=predicted_class, 
                                   confidence=f"{confidence:.2f}")
    return render_template('index.html', image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
