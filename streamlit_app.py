import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Function to plot RGB histograms and analyze issues
def analyze_and_plot_histograms(image):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    results = []

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = hist.flatten()
        ax[i].plot(hist, color=col)
        ax[i].set_xlim([0, 255])
        ax[i].set_ylim([0, 255])

        shift_left, shift_right, shift_left_magnitude, shift_right_magnitude = detect_shift(hist)
        spectrum_issue = detect_spectrum_issue(hist)

        results.append({
            'shift_left': shift_left,
            'shift_right': shift_right,
            'shift_left_magnitude': shift_left_magnitude,
            'shift_right_magnitude': shift_right_magnitude,
            'spectrum_issue': spectrum_issue
        })

    st.pyplot(fig)
    
    # Use columns to display the analysis under each histogram
    cols = st.columns(3)
    for i, col in enumerate(color):
        result = results[i]
        with cols[i]:
            st.write(f"{col.upper()} Channel Analysis")
            st.write(f"Shift Left: {result['shift_left']} (Magnitude: {result['shift_left_magnitude']})")
            st.write(f"Shift Right: {result['shift_right']} (Magnitude: {result['shift_right_magnitude']})")
            st.write(f"Spectrum Issue: {result['spectrum_issue']}")
            st.write("")

    plot_3d_histogram(image)

# Function to detect shifts in the histogram
def detect_shift(hist):
    shift_left_magnitude = 0
    shift_right_magnitude = 0

    # Calculate left shift magnitude
    for i in range(len(hist)):
        if hist[i] > 3:
            break
        shift_left_magnitude += 1

    # Calculate right shift magnitude
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] > 3:
            break
        shift_right_magnitude += 1

    shift_left = shift_left_magnitude > 0
    shift_right = shift_right_magnitude > 0

    return shift_left, shift_right, shift_left_magnitude, shift_right_magnitude

# Function to detect spectrum issues in the histogram
def detect_spectrum_issue(hist):
    low_threshold = 0.05 * np.max(hist)
    high_threshold = 0.95 * np.max(hist)
    low_spectrum = np.sum(hist < low_threshold)
    high_spectrum = np.sum(hist > high_threshold)
    if low_spectrum > 0.5 * len(hist):
        return "Underexposure"
    elif high_spectrum > 0.5 * len(hist):
        return "Overexposure"
    else:
        return "None"

# Function to plot 3D histogram
def plot_3d_histogram(image):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the histogram for each channel
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        xs = np.arange(256)
        ys = hist

        ax.bar(xs, ys, zs=i, zdir='y', color=color, alpha=0.8, edgecolor=color)

    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Color Channel')
    ax.set_zlabel('Frequency')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Blue', 'Green', 'Red'])

    st.pyplot(fig)

def main():
    st.title("Image Histogram Analysis App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.header("RGB Histograms and Analysis")
        analyze_and_plot_histograms(image)

if __name__ == "__main__":
    main()
