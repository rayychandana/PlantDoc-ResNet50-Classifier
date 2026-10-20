
import json
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ---------------- Model configuration ----------------
MODEL_PATH = Path(__file__).parent / "PlantDoc_ResNet50_Final.keras"
LABELS_PATH = Path(__file__).parent / "class_indices.json"

MODEL_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1dtyHK7eaaooumZjSM0ymBvc0p1Pp5k3Z"
)


# ---------------- Download model ----------------
def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    st.info("Downloading the trained model. The first run may take a few minutes.")

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    downloaded_path = gdown.download(
        MODEL_URL,
        str(MODEL_PATH),
        quiet=False,
        fuzzy=True
    )

    if (
        not downloaded_path
        or not MODEL_PATH.exists()
        or MODEL_PATH.stat().st_size == 0
    ):
        raise RuntimeError(
            "Model download failed. Check that the Google Drive file "
            "is shared with Anyone with the link."
        )


# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- Load class labels ----------------
@st.cache_data
def load_class_names():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "class_indices.json is missing from the GitHub repository."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        class_indices = json.load(file)

    return [
        name
        for name, index in sorted(
            class_indices.items(),
            key=lambda item: item[1]
        )
    ]


# ---------------- App interface ----------------
st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image to predict its disease "
    "using a fine-tuned ResNet50 model."
)

st.info(
    "This is an experimental machine-learning tool. "
    "Predictions may be incorrect and should be verified."
)


# ---------------- Image upload ----------------
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Prediction")

            if st.button("🔍 Detect Disease", type="primary"):
                if not LABELS_PATH.exists():
                    st.error(
                        "class_indices.json is missing from the repository."
                    )
                else:
                    with st.spinner(
                        "Loading the model and analyzing the leaf..."
                    ):
                        model = load_model()
                        class_names = load_class_names()

                        image_resized = image.resize((224, 224))
                        image_array = np.asarray(
                            image_resized,
                            dtype=np.float32
                        )

                        image_array = preprocess_input(image_array)
                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        predictions = model.predict(
                            image_array,
                            verbose=0
                        )[0]

                    if len(predictions) != len(class_names):
                        st.error(
                            "The model output does not match "
                            "the number of class labels."
                        )
                    else:
                        top_indices = np.argsort(predictions)[::-1][:5]
                        best_index = int(top_indices[0])

                        st.success(
                            f"Predicted class: {class_names[best_index]}"
                        )

                        st.write("Top 5 model scores")

                        results = {
                            class_names[int(i)]: float(predictions[i])
                            for i in top_indices
                        }

                        st.bar_chart(results)

                        for index in top_indices:
                            st.write(
                                f"**{class_names[int(index)]}:** "
                                f"{float(predictions[index]):.2%}"
                            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

        with st.expander("Technical details"):
            st.code(str(error))


# ---------------- Footer ----------------
st.divider()

st.caption(
    "PlantDoc dataset | ResNet50 | TensorFlow and Streamlit"
)
