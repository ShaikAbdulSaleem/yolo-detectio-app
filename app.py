# app.py
import os
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_FOLDER = 'static/annotated'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Load YOLO model (make sure yolov8n.pt is in project folder or specify full path)
model = YOLO('yolov8n.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run YOLO detection
    results = model(filepath)

    # Save annotated image
    annotated_img_path = os.path.join(ANNOTATED_FOLDER, 'annotated.jpg')
    results[0].save(filename=annotated_img_path)

    # Pass annotated image path to result page
    return render_template('result.html', image_path=annotated_img_path)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    # For local testing only; on Render use gunicorn to run
    app.run(host='0.0.0.0', port=10000)
