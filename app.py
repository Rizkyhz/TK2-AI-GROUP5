
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

CLASS_NAMES = [
    "Pesawat", "Mobil", "Burung", "Kucing", "Rusa",
    "Anjing", "Katak", "Kuda", "Kapal", "Truk"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_optimized.keras")

st.title("Klasifikasi Gambar dengan CNN")
st.write(
    "Unggah gambar untuk memperoleh prediksi "
    "menggunakan model CNN yang dilatih dengan CIFAR-10."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("Prediksi Gambar", type="primary"):
        # CIFAR-10 menggunakan input 32 x 32 RGB
        resized_image = image.resize((32, 32))
        image_array = np.array(
            resized_image,
            dtype=np.float32
        ) / 255.0

        image_batch = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(
            image_batch,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        st.subheader("Hasil Prediksi")
        st.success(
            f"Kelas: {CLASS_NAMES[predicted_index]}"
        )
        st.metric(
            "Skor keyakinan model",
            f"{confidence * 100:.2f}%"
        )

        st.caption(
            "Catatan: model dilatih pada gambar CIFAR-10 "
            "berukuran 32 x 32 piksel. Foto nyata yang "
            "kompleks mungkin tidak dikenali dengan tepat."
        )

        st.subheader("Skor untuk Setiap Kelas")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{name}: {probabilities[i] * 100:.2f}%")
            st.progress(float(probabilities[i]))
