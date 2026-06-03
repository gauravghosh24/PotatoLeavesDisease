from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

model = load_model("plant_disease_model.keras")

CLASS_NAMES = [
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy"
]

def predict_image(img):

    img = img.resize((256,256))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    idx = np.argmax(prediction)

    confidence = round(
        100*np.max(prediction),2
    )

    return CLASS_NAMES[idx], confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file)

    disease,confidence = predict_image(img)

    return render_template(
        "index.html",
        disease=disease,
        confidence=confidence
    )

if __name__=="__main__":
    app.run(debug=True)