from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image
import os

UPLOAD_FOLDER = os.path.abspath('../Front-end/images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load Hugging Face image classifier
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Load and classify the image
        img = Image.open(filepath).convert("RGB")
        result = classifier(img)
        top_result = result[0]
        score = round(top_result['score'] * 100, 2)
        label = top_result['label']

        return jsonify({
            'filename': filename,
            'score': score,
            'label': label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
