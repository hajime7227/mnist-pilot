import io

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.title("MNIST手書き数字認識")

uploaded_file = st.file_uploader(
    "手書き数字の画像をアップロードしてください", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    response = requests.post(API_URL, files={"file": img_bytes})

    if response.status_code == 200:
        predicted_digit = response.json()["predicted_digit"]
        st.write(f"### 予測結果: {predicted_digit}")
    else:
        st.write("予測中にエラーが発生しました")
