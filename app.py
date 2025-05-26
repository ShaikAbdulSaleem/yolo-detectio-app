!pip install ultralytics
# app.py - Backend with Flask
from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_FOLDER = 'static/annotated'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

model = YOLO('yolov8n.pt')  # Make sure this file is in your project or correctly referenced

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    image = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    # Detection
    results = model(path)
    annotated_path = os.path.join(ANNOTATED_FOLDER, 'annotated.jpg')
    results[0].save(filename=annotated_path)

    return render_template('result.html', image_path=annotated_path)

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
