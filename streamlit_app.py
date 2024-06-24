import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Function to plot RGB histograms and analyze issues
def analyze_and_plot_histograms(image, corrected=False, sliders=None):
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

        if corrected and sliders:
            ax[i].axvline(x=sliders[i][0], color='black', linestyle='--')
            ax[i].axvline(x=sliders[i][1], color='black', linestyle='--')

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

    return results

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

# Function to apply curve adjustments based on slider values
def apply_curve_adjustments(image, sliders):
    adjusted_image = image.copy()
    color = ('b', 'g', 'r')

    for i in range(3):
        left_val, right_val = sliders[i]
        adjusted_image[:, :, i] = np.clip(np.interp(image[:, :, i], [0, left_val, right_val, 255], [0, 0, 255, 255]), 0, 255)

    return adjusted_image

def main():
    st.title("Image Histogram Adjustment App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.header("RGB Histograms and Analysis")
        results = analyze_and_plot_histograms(image)

        # Display sliders for each channel
        st.header("Adjust RGB Curves")
        sliders = []
        for i, col in enumerate(('R', 'G', 'B')):
            left_val = st.slider(f'{col} Channel Left Value', 0, 255, results[i]['shift_left_value'] or 0)
            right_val = st.slider(f'{col} Channel Right Value', 0, 255, results[i]['shift_right_value'] or 255)
            sliders.append((left_val, right_val))

        if st.button('Apply Adjustments'):
            adjusted_image = apply_curve_adjustments(image, sliders)
            st.image(adjusted_image, caption='Adjusted Image', use_column_width=True)
            st.header("Adjusted RGB Histograms and Analysis")
            analyze_and_plot_histograms(adjusted_image, corrected=True, sliders=sliders)

if __name__ == "__main__":
    main()
