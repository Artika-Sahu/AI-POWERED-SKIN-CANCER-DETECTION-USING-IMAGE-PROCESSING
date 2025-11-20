# AI-POWERED-SKIN-CANCER-DETECTION-USING-IMAGE-PROCESSING
A deep learning–based skin cancer detection system using CNN and Keras. The model classifies dermoscopic images as benign or malignant with high accuracy. Includes data preprocessing, model training, evaluation metrics, and a Flask web app for real-time image upload and prediction.
REQUIRMENTS
tensorflow
keras
numpy
pandas
matplotlib
opencv-python
flask
scikit-learn
Pillow
FOLDER STRUCTURE
skin-cancer-detection/
│
├── app.py
├── model_training.ipynb
├── testing.ipynb
├── model.h5                    
├── requirements.txt
├── README.md
│
├── static/
│   ├── style.css
│   └── uploaded_images/
│
├── templates/
│   └── index.html
│
├── screenshots/
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   ├── confusion_matrix.png
│   └── sample_prediction.png
│
└── .gitignore
GITIGNORE FILE
__pycache__/
*.pyc
*.pkl
*.h5
*.csv
*.json
dataset/
*.DS_Store
venv/
.env
