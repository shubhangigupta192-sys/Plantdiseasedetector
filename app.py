# app.py - the Plant Disease Detection website
# Run it with:   streamlit run app.py

import pickle
import warnings
import numpy as np
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Plant Disease Detection", page_icon="\U0001F33F")


# ---------- 1. load the model we saved in the notebook ----------
@st.cache_resource
def load_model():
    with open("plant_disease_model.pkl", "rb") as f:
        return pickle.load(f)


saved = load_model()
IMG_SIZE = saved["img_size"]
BINS = saved["bins"]


# ---------- 2. the same feature engineering as the notebook ----------
def make_features(picture):
    total_pixels = IMG_SIZE * IMG_SIZE

    histogram = []
    for colour in range(3):
        counts, _ = np.histogram(picture[:, :, colour], bins=BINS, range=(0, 1))
        histogram.append(counts / total_pixels)

    average_colour = picture.mean(axis=(0, 1))
    colour_spread = picture.std(axis=(0, 1))

    return np.concatenate(histogram + [average_colour, colour_spread])


# ---------- 3. treatment suggestions ----------
TIPS = {
    "scab": "Rake up fallen leaves, prune the tree for airflow, spray a fungicide at bud break.",
    "blight": "Remove the infected leaves, do not water the leaves, use a copper based fungicide.",
    "spot": "Prune the affected leaves, give the plants more space, avoid overhead watering.",
    "rust": "Clear away infected leaves, rotate the crop next season, spray a sulphur fungicide.",
    "mildew": "Lower the humidity, space the plants further apart, spray a sulphur solution.",
    "mold": "Improve ventilation, lower the humidity, remove the affected leaves.",
    "virus": "There is no cure - remove the plant and control insects like whiteflies and aphids.",
    "mite": "Wash the leaves with water, then use insecticidal soap or neem oil.",
    "scorch": "Water deeply and regularly, mulch the soil, avoid too much fertiliser.",
    "measles": "Prune the infected wood in dry weather and seal the pruning wounds.",
    "rot": "Remove infected fruit and leaves, improve drainage, spray fungicide early in the season.",
    "haunglongbing": "Remove the infected tree and control the psyllid insect that spreads it.",
}


def get_tip(disease_name):
    for word in TIPS:
        if word in disease_name.lower():
            return TIPS[word]
    return "Keep the leaves dry, remove the damaged leaves and check the plant again in a few days."


# ---------- 4. the web page ----------
st.title("\U0001F33F Plant Disease Detection")
st.write("Upload a photo of a leaf and this app will tell you if the plant is sick.")
st.caption(f"Model: {saved['name']}  |  accuracy: {saved['test_accuracy'] * 100:.1f}%"
           f"  |  {len(saved['classes'])} plant diseases")

uploaded_file = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload a leaf photo to start.")
else:
    # show the photo the user uploaded
    image = Image.open(uploaded_file)
    st.image(image, caption="Your leaf", width=320)

    # prepare the photo exactly like the training photos
    photo = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    picture = np.array(photo, dtype=np.float32) / 255.0
    features = make_features(picture)

    # ask the model
    probabilities = saved["model"].predict_proba([features])[0]
    best = int(np.argmax(probabilities))
    full_name = saved["classes"][best]
    confidence = probabilities[best] * 100

    # the names look like "Tomato___Late_blight"
    plant_name, _, disease_name = full_name.partition("___")
    plant_name = plant_name.replace("_", " ").strip()
    disease_name = disease_name.replace("_", " ").strip()

    # show the result
    st.subheader("Result")
    if "healthy" in full_name.lower():
        st.success(f"This {plant_name} leaf is HEALTHY \u2705")
    else:
        st.error(f"This {plant_name} leaf is DISEASED \u26A0\ufe0f")
        st.write(f"**Disease found:** {disease_name}")
        st.info(f"**What to do:** {get_tip(full_name)}")

    st.write(f"**Confidence:** {confidence:.1f}%")
    st.progress(min(int(confidence), 100))

    with st.expander("Show the top 5 guesses"):
        top5 = np.argsort(probabilities)[::-1][:5]
        for i in top5:
            name = saved["classes"][i].replace("___", " - ").replace("_", " ")
            st.write(f"{name} - {probabilities[i] * 100:.1f}%")
