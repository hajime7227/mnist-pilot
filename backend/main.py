import numpy as np
from backend.predict import predict_digit
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()


@app.post("/predict")
async def predict_mnist(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L").resize((28, 28))
    image_array = np.array(image)
    predicted_digit = predict_digit(image_array)
    return {"predicted_digit": int(predicted_digit)}
