import os
import gdown

MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_detector.h5")
FILE_ID = "1ZVQSMEZ6w0JUYzxV6qWCnVQJySnA6fiC"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print(f"Model downloaded to {MODEL_PATH}")
    else:
        print("Model already exists, skipping download.")

if __name__ == "__main__":
    download_model()
