
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")
os.makedirs("static", exist_ok=True)

def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, image)
    return output_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    input_path = "static/input.jpg"
    file.save(input_path)
    output_path = detect_objects(input_path)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
