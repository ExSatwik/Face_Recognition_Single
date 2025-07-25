import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
import shutil
import time
import scipy
import pickle

from scipy.spatial.distance import cosine, euclidean

#import streamlit as st
import base64

def set_bg_video(video_file):
    with open(video_file, "rb") as file:
        video_bytes = file.read()
        encoded = base64.b64encode(video_bytes).decode()

    video_html = f"""
    <video autoplay loop muted playsinline style="position: fixed; right: 0; bottom: 0; min-width:100%; min-height:100%; z-index:-1;">
        <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# Call this function at the top of your app
#set_bg_video("bg (1).mp4")

# Your app content here
#st.title("Face Recognition App")
#st.write("This is your app content on top of video background.")

import streamlit as st

def set_background_image(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
        encoded_img = f"data:image/jpg;base64,{base64.b64encode(img_bytes).decode()}"

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{encoded_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background
import base64
set_background_image("Bag.jpg")



st.set_page_config(page_title="Face Recognition")
st.title("Face Recognition with DeepFace")
st.image("D:/Model ML/Photo_1.jpg", width=300)
# ---- Input method selection ----
input_method = st.selectbox("Choose input method", ["Upload", "Webcam"])

# ---- Model and distance metric ----
model_name = st.selectbox("Choose model", ["VGG-Face", "Facenet", "ArcFace"])
distance_metric = st.selectbox("Choose distance metric", ["cosine", "euclidean", "euclidean_l2"])

# ---- Define the correct .pkl file to use ----
model_key = model_name.lower().replace("-", "").replace(" ", "")
embedding_file = f"./embeddings/representations_{model_key}.pkl"

# ---- Thresholds ----
thresholds = {
    "Facenet": {"cosine": 0.35, "euclidean": 8.5, "euclidean_l2": 0.85},
    "VGG-Face": {"cosine": 0.35, "euclidean": 12, "euclidean_l2": 1.1},
    "ArcFace": {"cosine": 0.35, "euclidean": 9.5, "euclidean_l2": 0.85}
}

threshold = thresholds.get(model_name, {}).get(distance_metric, 0.4)


# ---- Distance function ----
def calculate_distance(e1, e2, metric):
    if metric == "cosine":
        return cosine(e1, e2)
    elif metric == "euclidean":
        return euclidean(e1, e2)
    elif metric == "euclidean_l2":
        e1 = np.array(e1) / np.linalg.norm(e1)
        e2 = np.array(e2) / np.linalg.norm(e2)
        return euclidean(e1, e2)
    else:
        return None

# ---- Function to add unmatched image to DB ----
def offer_add_to_database(image_path):
    if st.checkbox("Do you want to add this face to the database?"):
        person_name = st.text_input("Enter name for this person (used as filename):")
        if st.button("Confirm and Add"):
            save_name = person_name.strip().lower().replace(" ", "_") + f"_{int(time.time())}.jpg"
            save_path = os.path.join("faces", save_name)
            shutil.copy(image_path, save_path)
            st.success(f"✅ Image saved as `{save_name}` in `faces/` folder.")

            # Recompute only for selected model
            with st.spinner("Updating database..."):
                new_reps = []
                for file in os.listdir("faces"):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join("faces", file)
                        try:
                            emb = DeepFace.represent(
                                img_path=img_path,
                                model_name=model_name,
                                detector_backend="opencv",
                                enforce_detection=True
                            )
                            for obj in emb:
                                obj["identity"] = img_path
                                new_reps.append(obj)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                pd.DataFrame(new_reps).to_pickle(embedding_file)
                st.success("✅ Database updated for model!")

# ---- Handle input method ----
query_img_path = None

if input_method == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            query_img_path = tmp_file.name
        st.image(query_img_path, caption="Uploaded Image", width=300)

elif input_method == "Webcam":
    captured_image = st.camera_input("📸 Capture a photo")
    if captured_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(captured_image.getvalue())
            query_img_path = tmp_file.name
        st.image(query_img_path, caption="Captured Image", width=300)

# ---- Run face recognition ----
if query_img_path:
    with st.spinner("Searching for match..."):
        # Load embeddings
        try:
            print("checkpt1")
            df = pd.read_pickle(embedding_file)
            print(df)
        except Exception as e:
            st.error(f"❌ Failed to load embeddings: {e}")
            print(df)
            st.stop()

        # Get query embedding
        try:
            query_embedding = DeepFace.represent(
                img_path=query_img_path,
                model_name=model_name,
                detector_backend="opencv",
                enforce_detection=True
            )[0]["embedding"]
        except Exception as e:
            st.error(f"❌ Failed to extract embedding: {e}")
            st.stop()

        distances = []
        for _, row in df.iterrows():
            db_embedding = row["embedding"]
            dist = calculate_distance(query_embedding, db_embedding, distance_metric)
            distances.append((row["identity"], dist))

        if distances:
            distances.sort(key=lambda x: x[1])
            identity, distance = distances[0]

            st.markdown(f"**Best Match:** `{os.path.basename(identity)}`")
            st.markdown(f"**Distance:** `{distance:.4f}` (Threshold: `{threshold}`)")

            if distance <= threshold:
                st.success("✅ Match Accepted")
                img = cv2.imread(identity)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Extract name from filename
                filename = os.path.basename(identity)
                name_only = os.path.splitext(filename)[0]

                # If dynamically saved with underscore and timestamp (e.g., Salman_Khan_1719573551)
                if "_" in name_only and name_only.split("_")[-1].isdigit():
                    name_only = " ".join(name_only.split("_")[:-1])

                # Capitalize name nicely
                person_name = name_only.replace("_", " ").title()

                # Show matched image with name
                st.markdown(f"### 🧑 Matched Person: **{person_name}**")
                st.image(img, width=300)

            else:
                st.error("❌ Match Rejected")
                offer_add_to_database(query_img_path)
        else:
            st.warning("No matches found.")
            offer_add_to_database(query_img_path)

    os.remove(query_img_path)
