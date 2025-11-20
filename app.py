from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = os.path.join('model', 'skin_cancer_model.keras')
model = load_model(MODEL_PATH)

# Define class names based on your dataset
class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # ✅ Ensure the static folder exists
            upload_folder = os.path.join(app.root_path, 'static')
            os.makedirs(upload_folder, exist_ok=True)

            # ✅ Save the uploaded file safely
            img_filename = file.filename
            img_path = os.path.join(upload_folder, img_filename)
            file.save(img_path)

            # ✅ Preprocess image for model
            img = image.load_img(img_path, target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # ✅ Make prediction
            preds = model.predict(x)
            pred_class = class_names[np.argmax(preds[0])]
            confidence = np.max(preds[0]) * 100

            prediction = f"Prediction: {pred_class} ({confidence:.2f}% confidence)"

    return render_template('index.html', prediction=prediction, img_filename=img_filename)


if __name__ == '__main__':
    app.run(debug=True)
