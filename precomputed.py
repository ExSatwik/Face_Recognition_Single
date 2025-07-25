from deepface import DeepFace
import pandas as pd
import os

# ✅ All models you want to support
models = ["Facenet", "VGG-Face", "ArcFace"]

# ✅ Your dataset folder (flat directory of face images)
db_path = "faces"  # Make sure this exists

# Loop through all models
for model in models:
    print(f"\n🔁 Generating embeddings for model: {model}")

    embeddings = []
    for file in os.listdir(db_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(db_path, file)
            try:
                # 🔍 Represent image using selected model
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=model,
                    detector_backend="opencv",  # ✅ Detect face automatically
                    enforce_detection=True
                )

                for rep in reps:
                    rep["identity"] = img_path
                    embeddings.append(rep)

            except Exception as e:
                print(f"⚠️ Error with {file}: {e}")

    # Save to file named by model
    out_name = f"embeddings/representations_{model.lower().replace('-', '').replace(' ', '')}.pkl"
    pd.DataFrame(embeddings).to_pickle(out_name)
    print(f"✅ Saved embeddings to {out_name}")
