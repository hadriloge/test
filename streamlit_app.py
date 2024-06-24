import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageEnhance

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Function to plot RGB histograms and analyze issues
def analyze_and_plot_histograms(image, corrected=False):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    results = []

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = hist.flatten()
        ax[i].plot(hist, color=col)
        ax[i].set_xlim([0, 255])
        ax[i].set_ylim([0, np.max(hist)])  # Set y-axis to the max of the histogram

        shift_left_value, shift_right_value = detect_shift(hist)
        spectrum_issue = detect_spectrum_issue(hist)

        results.append({
            'shift_left_value': shift_left_value,
            'shift_right_value': shift_right_value,
            'spectrum_issue': spectrum_issue
        })

        if corrected:
            ax[i].axvline(x=shift_left_value, color='black', linestyle='--')
            ax[i].axvline(x=shift_right_value, color='black', linestyle='--')

    st.pyplot(fig)
    
    # Use columns to display the analysis under each histogram
    cols = st.columns(3)
    for i, col in enumerate(color):
        result = results[i]
        with cols[i]:
            st.write(f"{col.upper()} Channel Analysis")
            st.write(f"First Significant Left Value: {result['shift_left_value']}")
            st.write(f"First Significant Right Value: {result['shift_right_value']}")
            st.write(f"Spectrum Issue: {result['spectrum_issue']}")
            st.write("")

    plot_3d_histogram(image, results)

# Function to detect shifts in the histogram
def detect_shift(hist):
    shift_left_value = None
    shift_right_value = None

    # Find the first significant left value
    for i in range(len(hist)):
        if hist[i] > 3000:
            shift_left_value = i
            break

    # Find the first significant right value
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] > 3000:
            shift_right_value = i
            break

    return shift_left_value, shift_right_value

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

# Function to correct RGB values
def correct_rgb(image):
    corrected_image = image.copy()
    color = ('b', 'g', 'r')

    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        shift_left_value, shift_right_value = detect_shift(hist)

        # Apply corrections based on the shift values
        corrected_image[:, :, i] = np.clip(np.interp(image[:, :, i], (0, shift_left_value), (shift_left_value, 255)), 0, 255)

    # Detect and correct over/underexposure
    spectrum_issue = detect_spectrum_issue(cv2.calcHist([corrected_image], [0, 1, 2], None, [256], [0, 256]).flatten())
    if spectrum_issue == "Underexposure":
        enhancer = ImageEnhance.Brightness(Image.fromarray(corrected_image))
        corrected_image = np.array(enhancer.enhance(1.2))
    elif spectrum_issue == "Overexposure":
        enhancer = ImageEnhance.Brightness(Image.fromarray(corrected_image))
        corrected_image = np.array(enhancer.enhance(0.8))

    return corrected_image

# Function to plot 3D histogram
def plot_3d_histogram(image, results):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the histogram for each channel
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        xs = np.arange(256)
        ys = hist

        ax.bar(xs, ys, zs=i, zdir='y', color=color, alpha=0.8, edgecolor=color)

        # Plot the first significant values
        shift_left_value = results[i]['shift_left_value']
        shift_right_value = results[i]['shift_right_value']
        if shift_left_value is not None:
            ax.scatter([shift_left_value], [i], [ys[shift_left_value]], color='k', marker='o')
        if shift_right_value is not None:
            ax.scatter([shift_right_value], [i], [ys[shift_right_value]], color='k', marker='o')

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

        if st.button('Correct RGB'):
            corrected_image = correct_rgb(image)
            st.image(corrected_image, caption='Corrected Image', use_column_width=True)
            st.header("Corrected RGB Histograms and Analysis")
            analyze_and_plot_histograms(corrected_image, corrected=True)

if __name__ == "__main__":
    main()
