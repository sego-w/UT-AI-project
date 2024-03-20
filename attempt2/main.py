from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import base64
import io

app = Flask("imagerecog", static_folder=os.path.abspath('attempt2/static'))

# Load the Keras model
model = keras.models.load_model('attempt2/keras_model.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the labels
print() 
class_names = open('attempt2/labels.txt', "r").readlines()

# Define the text for each class
class_text = {
    'klaaspakend': 'Loputa vajadusel kergelt, et ei määriks teisi pakendeid ja kotti. Eemadada korgid ja kaaned, sildid võivad jääda.',
    'taara': 'Viige pakend lähimasse taaraautomaati. Juhul, kui masin ei võta seda vastu, visake kohapealsesse prügikasti ära',
    'paberpakend': 'Voldi suured papist pakendid kokku või rebi tükkideks, nii võtavad nad vähem ruumi. Veendu, et materjal on puhas ja kuiv.',
    'biojaatmed': 'Biojäätmed pane konteinerisse lahtiselt, paberkotis või täielikult biolaguneva ja komposteeruva kotiga.',
    'vanapaber': 'Kogu paber ja kartong muudest jäätmetest eraldi ka siis, kui teie majal pole selleks konteinerit. Vanapaber pane konteinerisse lahtiselt.',
    'plastpakend': 'Kogu pakendi- ja toidujäätmed eraldi ja segaolmejäätmete hulk väheneb märgatavalt!',
}

@app.route('/')
def home():
    css_file = os.path.join('static', 'style.css')
    if os.path.exists(css_file):
        print(f"CSS file exists at: {os.path.abspath(css_file)}")
    else:
        print(f"CSS file not found at: {os.path.abspath(css_file)}")
    
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Manipulator</title>
        <link rel="stylesheet" href="/static/style.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                border-left: 5px solid #add8e6;
                border-right: 5px solid #add8e6;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fff;
            }
            h1 {
                color: #333;
                font-size: 28px;
                margin-bottom: 20px;
            }
            p {
                color: #666;
                font-size: 18px;
                margin-bottom: 20px;
            }
            form {
                margin-bottom: 20px;
            }
            input[type="file"] {
                margin-bottom: 10px;
            }
            input[type="submit"] {
                padding: 10px 20px;
                background-color: #add8e6;
                color: #fff;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            #result {
                font-size: 18px;
                line-height: 1.6;
            }
            #result img {
                max-width: 300px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Lae üles pilt leitud prügist</h1>
            <p>Rakendus ennustab, mis tõenäosusega on mis prügiga tegemist!</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*">
                <input type="submit" value="Hinda pilti">
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.querySelector('form').addEventListener('submit', function(e) {
                e.preventDefault();
                var formData = new FormData(this);
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    var resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = 'Pildil on: ' + data.class_name.split(' ')[1] + '<br>Tõenäosusega ' + data.confidence_score + '<br>' + data.class_text + '<br><img src="data:image/jpeg;base64,' + data.image + '">';
                });
            });
        </script>
        
        <p>Kui sinu lähedal ei ole prügikonteinerit, saad leida lähima veebilehelt</p><a href=https://kuhuviia.ee>kuhuviia.ee</a>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    
    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Make predictions using the Keras model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])
    
    # Get the text for the predicted class
    predicted_text = class_text.get(class_name.split(' ')[1], 'Unknown class')
    
    # Convert the uploaded image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Return the prediction result as JSON
    result = {
        'class_name': class_name,
        'confidence_score': confidence_score,
        'class_text': predicted_text,
        'image': image_base64
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)