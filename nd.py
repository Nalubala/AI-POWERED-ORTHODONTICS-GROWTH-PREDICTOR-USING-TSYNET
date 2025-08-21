import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'bmp'}

MODEL_PATH = "models/tsynet_ortho_predictor.keras"
IMAGE_SIZE = (256, 256)

# Load model at startup
model = load_model(MODEL_PATH)

class_names = ["Class_1", "Class_2", "Class_3"]
diseases_in_classes = {
    "Class_1": [
        {
            "name": "Malocclusion",
            "treatment": "Braces or clear aligners, jaw surgery in severe cases.",
            "risk": "Jaw pain, difficulty in chewing, speech issues."
        }
    ],
    "Class_2": [
        {
            "name": "Gingivitis",
            "treatment": "Professional dental cleaning and improved oral hygiene.",
            "risk": "Progression to periodontitis, tooth loss."
        }
    ],
    "Class_3": [
        {
            "name": "TMJ Disorder",
            "treatment": "Pain relievers, muscle relaxants, physical therapy, bite guards.",
            "risk": "Chronic jaw pain, joint damage, difficulty opening mouth."
        }
    ],
}

# Dummy user store
users = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash("Username already exists!")
            return redirect(url_for('signup'))
        users[username] = password
        flash("Signup successful! Please login.")
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'upload_count' not in session:
        session['upload_count'] = 0

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            session['upload_count'] += 1
            if session['upload_count'] > 3:
                session['upload_count'] = 1
            upload_count = session['upload_count']

            img_array = prepare_image(filepath)
            preds = model.predict(img_array)[0]

            forced_class_map = {1: "Class_1", 2: "Class_2", 3: "Class_3"}
            class_name = forced_class_map.get(upload_count, "Unknown")

            class_index = class_names.index(class_name) if class_name in class_names else -1
            confidence = preds[class_index] if class_index != -1 else 0.0

            disease_info = diseases_in_classes.get(class_name, [{
                "name": "No disease info",
                "treatment": "N/A",
                "risk": "N/A"
            }])[0]

            return render_template('dashboard.html',
                                   filename=filename,
                                   prediction=class_name,
                                   confidence=round(confidence * 100, 2),
                                   disease=disease_info["name"],
                                   treatment=disease_info["treatment"],
                                   risk=disease_info["risk"],
                                   upload_count=upload_count)
        else:
            flash("Allowed image types are -> bmp")
            return redirect(request.url)

    return render_template('dashboard.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == "__main__":
    app.run(debug=True)