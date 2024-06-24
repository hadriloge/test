import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Function to plot RGB histograms
def plot_rgb_histograms(image):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax[i].plot(hist, color=col)
        ax[i].set_xlim([0, 255])
    st.pyplot(fig)

# Function to adjust the image based on histograms
def adjust_image(image):
    image = image.astype(np.float32) / 255.0
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        if hist[0] > 0.05 * sum(hist) or hist[-1] > 0.05 * sum(hist):
            image[..., i] = np.clip(image[..., i], 0.05, 0.95)
        image[..., i] = (image[..., i] - np.min(image[..., i])) / (np.max(image[..., i]) - np.min(image[..., i]))
    adjusted_image = (image * 255).astype(np.uint8)
    return adjusted_image

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def main():
    st.title("Image Editing App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.header("Original RGB Histograms")
        plot_rgb_histograms(image)

        if st.button("Apply Correction"):
            corrected_image = adjust_image(image)
            st.image(corrected_image, caption='Corrected Image', use_column_width=True)
            
            st.header("Corrected RGB Histograms")
            plot_rgb_histograms(corrected_image)

if __name__ == "__main__":
    main()
