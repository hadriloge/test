import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import tempfile

def plot_rgb_histograms(image):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax[i].plot(hist, color=col)
        ax[i].set_xlim([0, 256])
    st.pyplot(fig)

def correct_image(image):
    image = image.astype(np.float32) / 255.0
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        if hist[0] > 0.05 * sum(hist) or hist[-1] > 0.05 * sum(hist):
            image[..., i] = np.clip(image[..., i], 0.05, 0.95)
        image[..., i] = (image[..., i] - np.min(image[..., i])) / (np.max(image[..., i]) - np.min(image[..., i]))
    corrected_image = (image * 255).astype(np.uint8)
    return corrected_image

def main():
    st.title("Image Color Correction App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        image = Image.open(temp_file_path)
        image = np.array(image)

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        st.subheader("Original RGB Histograms")
        plot_rgb_histograms(image)

        corrected_image = correct_image(image)

        st.subheader("Corrected Image")
        st.image(corrected_image, use_column_width=True)

        st.subheader("Corrected RGB Histograms")
        plot_rgb_histograms(corrected_image)

        # Clean up the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
