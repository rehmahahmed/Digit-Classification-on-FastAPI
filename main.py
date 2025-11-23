import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # 1. Import StaticFiles
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = "mnist_model.h5"
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# --- Initialize App & Load Model ---
app = FastAPI()

# 2. Mount the "static" folder so the browser can access the image
# This tells FastAPI: "If a URL starts with /static, look in the 'static' folder"
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("MNIST Model loaded successfully.")
except Exception:
    print("Model not found. Please run train_model.py first.")

# --- Helper: Smart Image Processing ---
def transform_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("L")
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    target_size = (28, 28)
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("L", target_size, 0)
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    img_array = np.array(new_img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# --- HTML Frontend ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digit Classifier</title>
    <style>
        :root {
            --col-dark: #1A2A4F;
            --col-pink: #F7A5A5;
            --col-peach: #FFDBB6;
            --col-light: #FFF2EF;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--col-light);
            color: var(--col-dark);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        
        /* Header */
        header {
            background-color: var(--col-dark);
            color: var(--col-peach);
            padding: 0.8rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-radius:22px;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
        }
        
        /* Logo Image Styling */
        .logo-img {
            height: 80px; /* Adjust height as needed */
            width: auto;
            display: block;
        }

        .btn-project {
            background-color: var(--col-pink);
            color: var(--col-dark);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: bold;
            transition: background 0.3s;
        }
        .btn-project:hover {
            background-color: var(--col-peach);
        }

        /* Main Content */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .card {
            background-color: white;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            text-align: center;
            max-width: 450px;
            width: 100%;
            border-top: 6px solid var(--col-peach);
        }
        h1 { margin-top: 0; margin-bottom: 1.5rem; font-size: 2rem;}
        
        /* Upload Section */
        .upload-area {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 2px dashed var(--col-peach);
            background-color: #fafafa;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            min-height: 180px;
        }
        .upload-area:hover {
            background-color: #fff0e6;
            border-color: var(--col-pink);
        }
        
        .upload-text {
            margin: 0;
            font-size: 1.1rem;
            color: #666;
        }
        
        input[type="file"] { display: none; }
        
        #preview {
            max-width: 120px;
            max-height: 120px;
            margin-top: 15px;
            display: none;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 2px solid var(--col-dark);
        }

        .btn-classify {
            background-color: var(--col-dark);
            color: var(--col-light);
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: opacity 0.2s;
        }
        .btn-classify:hover { opacity: 0.9; }

        #result {
            margin-top: 2rem;
            font-size: 1.2rem;
            font-weight: bold;
            color: var(--col-dark);
            min-height: 1.5em;
            padding: 10px;
            border-radius: 5px;
        }

        /* Footer */
        footer {
            background-color: var(--col-dark);
            color: var(--col-peach);
            padding: 1.5rem 2rem;
            margin-top: auto;
            border-radius:22px;
        }
        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 900px;
            margin: 0 auto;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .col { font-size: 0.9rem; }
        .col a { color: var(--col-pink); text-decoration: none; transition: color 0.2s;}
        .col a:hover { color: #fff; }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <img src="/static/Rlogo - Digit.png" alt="Rlogo" class="logo-img">
        </div>
        <a href="https://rehmahprojects.com/projects.html" class="btn-project">More Projects</a>
    </header>

    <main>
        <div class="card">
            <h1>Handwritten Digit AI</h1>
            
            <label for="fileInput" class="upload-area">
                <span class="upload-text">📂 Click to Upload Image</span>
                <img id="preview" alt="Preview">
            </label>
            <input type="file" id="fileInput" accept="image/*">
            
            <button class="btn-classify" onclick="classifyImage()">Identify Digit</button>
            
            <div id="result"></div>
        </div>
    </main>

    <footer>
        <div class="footer-content">
            <div class="col">Project powered by <strong>rehmahprojects.com</strong></div>
            <div class="col">Contact: <a href="mailto:admin@rehmahprojects.com">admin@rehmahprojects.com</a></div>
        </div>
    </footer>

    <script>
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const uploadText = document.querySelector('.upload-text');
        const resultDiv = document.getElementById('result');

        // Show image preview
        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    uploadText.style.display = 'none'; 
                    resultDiv.innerText = '';
                };
                reader.readAsDataURL(file);
            }
        });

        async function classifyImage() {
            const file = fileInput.files[0];
            if (!file) {
                resultDiv.innerText = "⚠️ Please select an image first.";
                resultDiv.style.color = "#d9534f";
                return;
            }

            resultDiv.style.color = "var(--col-dark)";
            resultDiv.innerText = "Thinking...";
            
            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();
                if (data.class) {
                    resultDiv.innerHTML = `Prediction: <span style="color: #F7A5A5; font-size: 2.5rem; display:block; margin-top:10px;">${data.class}</span>`;
                } else {
                    resultDiv.innerText = "Could not classify.";
                }
            } catch (error) {
                console.error(error);
                resultDiv.innerText = "Server Error.";
            }
        }
    </script>
</body>
</html>
"""

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    img_array = transform_image(image_bytes)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {"class": predicted_label, "confidence": confidence}