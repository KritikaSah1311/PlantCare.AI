import os
import sys
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Dynamically get the directory where app.py is stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up the uploads folder using the absolute path
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- BULLETPROOF MODEL LOADER ---
MODEL_PATH = os.path.join(BASE_DIR, 'mobilenetv2_best.keras')
ZIP_PATH = os.path.join(BASE_DIR, 'mobilenetv2_best.keras.zip')

print("\n" + "="*50)
print("🔍 Checking for model file...")

if not os.path.exists(MODEL_PATH):
    if os.path.exists(ZIP_PATH):
        print(f"🛠️ Auto-fixing: Found {ZIP_PATH}!")
        print("🛠️ Renaming it to remove the .zip extension...")
        os.rename(ZIP_PATH, MODEL_PATH)
    else:
        print("🚨 CRITICAL ERROR: Could not find the model file AT ALL.")
        print(f"🚨 I looked for: {MODEL_PATH}")
        sys.exit(1)

print(f"✅ Model file found at: {MODEL_PATH}")
print("⏳ Loading model into memory (this takes a few seconds)...")

try:
    model = load_model(MODEL_PATH)
    print("🚀 MODEL LOADED SUCCESSFULLY! Starting server...")
    print("="*50 + "\n")
except Exception as e:
    print(f"\n🚨 Keras failed to load the model! Error details:\n{e}")
    sys.exit(1)

# Define your class labels based on your training data
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    if hasattr(img, 'close'):
        img.close()
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    predicted_label = CLASS_NAMES[predicted_class_idx]
    
    parts = predicted_label.split("___")
    plant_type = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ")
    
    is_healthy = "healthy" in condition.lower()

    if is_healthy:
        actions = ["Maintain regular watering schedule", "Ensure adequate sunlight", "Monitor for any future changes"]
    else:
        actions = ["Isolate the affected plant", "Remove and destroy heavily infected leaves", f"Research specific treatments for {condition}"]

    return {
        "plant_type": plant_type,
        "condition": condition,
        "is_healthy": is_healthy,
        "confidence": confidence,
        "actions": actions
    }

# --- NEW MULTI-PAGE ROUTES ---
@app.route('/', methods=['GET'])
@app.route('/home.html', methods=['GET'])
def home():
    return send_file(os.path.join(BASE_DIR, 'home.html'))

@app.route('/about.html', methods=['GET'])
def about():
    return send_file(os.path.join(BASE_DIR, 'about.html'))

@app.route('/upload.html', methods=['GET'])
def upload():
    return send_file(os.path.join(BASE_DIR, 'upload.html'))

@app.route('/result.html', methods=['GET'])
def result():
    return send_file(os.path.join(BASE_DIR, 'result.html'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_disease(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)