import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Define your model
# -----------------------------
class ClassifierAutoencoder(nn.Module):
    def __init__(self):
        super(ClassifierAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=0), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        classification = self.classifier(latent)
        return reconstruction[:, :, :28, :28], classification

# -----------------------------
# Load trained weights
# -----------------------------
model = ClassifierAutoencoder()
model.load_state_dict(torch.load("mnist_autoencoder_classifier.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("MNIST Digit Classifier")
st.write("Draw a digit OR upload an image. The model will predict the digit and show probabilities.")

# --- Option 1: Draw on Canvas ---
st.subheader("Draw a digit (canvas)")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]  # grayscale
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()/255.0
    img = torch.nn.functional.interpolate(img, size=(28,28))  # downsample to 28x28
    img = 1.0 - img  # invert colors: MNIST is white digit on black background

    with torch.no_grad():
        _, preds = model(img)
        probs = torch.softmax(preds, dim=1).numpy()[0]

    st.bar_chart(probs)
    st.write(f"Predicted Digit: {np.argmax(probs)}")

# --- Option 2: Upload Image ---
st.subheader("Upload a digit image")
uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28,28))
    st.image(img, caption="Uploaded Image", width=150)

    img_tensor = torch.tensor(np.array(img)).unsqueeze(0).unsqueeze(0).float()/255.0
    img_tensor = 1.0 - img_tensor  # invert colors to match MNIST

    with torch.no_grad():
        _, preds = model(img_tensor)
        probs = torch.softmax(preds, dim=1).numpy()[0]

    st.bar_chart(probs)
    st.write(f"Predicted Digit: {np.argmax(probs)}")
