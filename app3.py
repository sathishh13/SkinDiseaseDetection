from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import os

app = Flask(__name__)
model = load_model('C:\\Users\\admin\\Desktop\\Skin predict\\Skindensenet1.h5')

@app.route('/')
def index_view():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_skin_condition(class_index):
    class_labels = {
        0: "Actinic keratosis",
        1: "Atopic dermatitis",
        2: "Benign keratosis",
        3: "Candidiasis Ringworm Tinea",
        4: "Dermatofibroma",
        5: "Melanocytic nevus",
        6: "Melonama",
        7: "Squamous cell carcinoma",
        8: "Vascular lesion"
    }
    return class_labels.get(class_index, "Unknown")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request"

    file = request.files['file']
    if file.filename == '':
        return "No image selected"

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('static/images', filename)
        file.save(file_path)
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        skin = get_skin_condition(class_index)
        return render_template('predict.html', skin=skin, user_image=file_path)
    else:
        return "Invalid file format. Please upload an image file (jpg, jpeg, png)"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
