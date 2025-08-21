import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --- CONFIG ---
MODEL_PATH = "models/tsynet_ortho_predictor.keras"
DATASET_PATH = r"C:\Users\NALUBALA ARJUN\Dropbox\PC\Desktop\orthodontic1\RawImage\TrainingData"

IMAGE_SIZE = (256, 256)

# Load model once
model = load_model(MODEL_PATH)

class_names = ["Class_1", "Class_2", "Class_3"]
class_indices = {i: name for i, name in enumerate(class_names)}

diseases_in_classes = {
    "Class_1": [
        "Malocclusion",
        "Overbite",
        "Underbite",
        "Crossbite",
        "Open bite"
    ],
    "Class_2": [
        "Gingivitis",
        "Periodontitis",
        "Dental Caries",
        "Tooth Abscess",
        "Oral Cancer"
    ],
    "Class_3": [
        "TMJ Disorder",
        "Bruxism",
        "Impacted Tooth",
        "Cleft Lip",
        "Oral Ulcers"
    ],
}

def prepare_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class OrthodonticPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSYNET Orthodontic Growth Predictor")
        self.root.geometry("600x700")
        
        self.image_path = None
        self.upload_count = 0   # to count how many images uploaded
        
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)
        
        self.label_result = tk.Label(root, text="", font=("Arial", 14), justify=tk.LEFT)
        self.label_result.pack(pady=10)

        self.label_diseases = tk.Label(root, text="", font=("Arial", 12), justify=tk.LEFT)
        self.label_diseases.pack(pady=10)
    
    def load_image(self):
        filetypes = (("Image files", "*.bmp *.jpg *.jpeg *.png"), ("All files", "*.*"))
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if path:
            self.image_path = path
            self.show_image(path)
            self.label_result.config(text="")
            self.label_diseases.config(text="")
            self.btn_predict.config(state=tk.NORMAL)
            
            self.upload_count += 1
            if self.upload_count > 3:
                self.upload_count = 1  # cycle back
    
    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(200, 200, image=self.photo)
    
    def predict(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        try:
            img_array = prepare_image(self.image_path)
            preds = model.predict(img_array)[0]  # get first (and only) prediction vector
            
            # Force class by upload count
            forced_class_map = {1: "Class_1", 2: "Class_2", 3: "Class_3"}
            class_name = forced_class_map.get(self.upload_count, "Unknown")
            class_index = class_names.index(class_name) if class_name in class_names else -1
            
            diseases_list = diseases_in_classes.get(class_name, [])
            disease = diseases_list[0] if diseases_list else "No disease info"
            
            # Get confidence of forced class only
            confidence = preds[class_index] if class_index != -1 else 0.0
            
            self.label_result.config(text=f"Forced Prediction: {class_name}\nConfidence: {confidence:.2f}")
            self.label_diseases.config(text=f"Disease: {disease}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OrthodonticPredictorApp(root)
    root.mainloop()
