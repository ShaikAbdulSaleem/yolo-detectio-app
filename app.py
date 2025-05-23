import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Folder to save uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your YOLO model once
model = YOLO("yolov8n.pt")  # or yolov8s.pt, etc.

# Home page: shows upload options
@app.route('/')
def home():
    return render_template('index.html')

# Handle image upload and detection
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save original image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Read image with OpenCV
    img = cv2.imread(img_path)
    if img is None:
        return "Failed to read image", 400

    # Run YOLO detection
    results = model(img)

    # Annotate image
    annotated_img = results[0].plot()  # Returns numpy array with boxes

    # Save annotated image
    result_filename = 'result_' + file.filename
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, annotated_img)

    # Redirect to result page showing this image
    return redirect(url_for('show_result', filename=result_filename))

# Handle video upload (basic save only)
@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video uploaded", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # TODO: Add video processing and detection here if you want

    return f"Video uploaded successfully: {file.filename}"

# Webcam page placeholder
@app.route('/webcam')
def webcam():
    return "<h2>Webcam streaming not yet implemented. Coming soon! 🎥</h2><a href='/'>Back</a>"

# Show result page with annotated image
@app.route('/result/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)

# Serve uploaded and result files (optional if you want direct access)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
