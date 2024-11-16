import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import cv2
from werkzeug.utils import secure_filename


# Initialize Flask app
app = Flask(__name__)

# Set the upload folder for saving the image
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and label encoder
model = tf.keras.models.load_model('conv2d.h5')  # or 'conv2d.h5'
model = tf.keras.models.load_model('conv2d.keras')  # or 'conv2d.h5'
with open('conv2d_lb.pickle', 'rb') as f:
    lb = pickle.load(f)

# Image processing function
def prepare_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def prepare_image1(image_file):
    image_file.seek(0)

    # Convert the uploaded file to a NumPy array compatible with cv2
    file_bytes = np.frombuffer(image_file.read(), np.uint8)

    # Decode the image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode the uploaded image. Ensure the file is a valid image.")

    # Resize the image to match the model's input size
    img = cv2.resize(img, (128, 128))

    # Scale pixel values to [0, 1]
    img = img.astype('float32') / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img, axis=0)

    return img_array

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')  # This will render the HTML page

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is part of the request
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    # Check if the file is empty
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Securely save the uploaded image
    filename = secure_filename(image_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(file_path)

    # Prepare image for prediction
    image = prepare_image1(image_file)
    
    # Predict using the model
    preds = model.predict(image)
    max_label = preds.argmax(axis=1)[0]  # Get the class with the highest probability
    label = lb.classes_[max_label]  # Get the class label from the label encoder

    # Check if confidence is greater than 0.9
    confidence = np.max(preds)
    if confidence >= 0.9:
        prediction_text = "Tiger Detected"
    else:
        prediction_text = "Not a Tiger"

    # Return the prediction result along with image URL
    return jsonify({
        'prediction': prediction_text,
        'confidence': str(confidence),
        'image_url': file_path  # Send back the URL to the uploaded image
    })

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
