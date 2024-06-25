import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

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

        shift_left_value, shift_right_value, significant_spikes = detect_shift(hist)
        spectrum_issue = detect_spectrum_issue(hist)

        results.append({
            'shift_left_value': shift_left_value,
            'shift_right_value': shift_right_value,
            'spectrum_issue': spectrum_issue,
            'significant_spikes': significant_spikes
        })

    st.pyplot(fig)

    # Display analysis
    cols = st.columns(3)
    for i, col in enumerate(color):
        result = results[i]
        with cols[i]:
            st.write(f"{col.upper()} Channel Analysis")
            st.write(f"First Significant Left Value: {result['shift_left_value']}")
            st.write(f"First Significant Right Value: {result['shift_right_value']}")
            st.write(f"Spectrum Issue: {result['spectrum_issue']}")
            st.write(f"Significant Spikes: {result['significant_spikes']}")
            st.write("")

    return results

# Function to detect shifts and significant spikes in the histogram
def detect_shift(hist):
    shift_left_value = None
    shift_right_value = None
    significant_spikes = []

    for i in range(len(hist)):
        if hist[i] > 3000:
            shift_left_value = i
            break

    for i in range(len(hist) - 1, -1, -1):
        if hist[i] > 3000 and i != 255:
            shift_right_value = i
            break

    threshold = 0.1 * np.max(hist)
    for i in range(1, 25):
        if abs(hist[i] - hist[i - 1]) > threshold and hist[i] > 3000:
            significant_spikes.append((i, hist[i]))
    for i in range(len(hist) - 25, len(hist) - 1):
        if abs(hist[i] - hist[i - 1]) > threshold and hist[i] > 3000:
            significant_spikes.append((i, hist[i]))

    return shift_left_value, shift_right_value, significant_spikes

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

# Function for histogram equalization
def equalize_histogram(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# Function for gamma correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function for white balance correction
def white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# Function to apply curve adjustments based on slider values
def apply_curve_adjustments(image, sliders):
    adjusted_image = image.copy()
    color = ('b', 'g', 'r')

    for i in range(3):
        left_val, right_val = sliders[i]
        adjusted_image[:, :, i] = np.clip(np.interp(image[:, :, i], [0, left_val, right_val, 255], [0, 0, 255, 255]), 0, 255)

    return adjusted_image

# Function to apply extra enhancements (sharpening and contrast)
def apply_extra_enhancements(image, sharpness_factor, contrast_factor):
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharpened_image = enhancer.enhance(sharpness_factor)
    enhancer = ImageEnhance.Contrast(sharpened_image)
    contrasted_image = enhancer.enhance(contrast_factor)
    return np.array(contrasted_image)

# Function to remove spikes in the histogram
def remove_spikes(image, results):
    adjusted_image = image.copy()
    color = ('b', 'g', 'r')

    for i in range(3):
        spikes = results[i]['significant_spikes']
        for spike in spikes:
            adjusted_image[:, :, i][adjusted_image[:, :, i] == spike[0]] = spike[0] - 1  # Shift the spike value to the left

    return adjusted_image

def main():
    st.set_page_config(layout="centered")
    st.title("Image Histogram Adjustment App")

    steps = ["Upload Image", "Analysis", "Remove Spikes", "Adjust Significant Values", "Equalize Histogram", "Gamma Correction", "White Balance", "Apply Extra Enhancements"]

    if "step" not in st.session_state:
        st.session_state.step = 0

    def next_step():
        if st.session_state.step < len(steps) - 1:
            st.session_state.step += 1

    def prev_step():
        if st.session_state.step > 0:
            st.session_state.step -= 1

    st.sidebar.title("Navigation")
    if st.sidebar.button("Previous Step"):
        prev_step()
    if st.sidebar.button("Next Step"):
        next_step()

    progress = st.sidebar.progress(st.session_state.step / (len(steps) - 1))

    # Step 1: Upload an image
    if st.session_state.step == 0:
        st.header("1. Choose an image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.session_state.image = load_image(uploaded_file)
            st.image(st.session_state.image, caption='Uploaded Image', use_column_width=True)

    if "image" in st.session_state:
        image = st.session_state.image

        # Step 2: Analysis
        if st.session_state.step == 1:
            st.header("2. RGB Histograms and Analysis")
            st.session_state.results = analyze_and_plot_histograms(image)

        # Step 3: Remove Spikes
        if st.session_state.step == 2:
            st.header("3. Remove Spikes")
            if st.button('Remove Spikes'):
                st.session_state.spike_removed_image = remove_spikes(image, st.session_state.results)
                st.image(st.session_state.spike_removed_image, caption='Spikes Removed Image', use_column_width=True)

        # Step 4: Adjust Significant Values
        if st.session_state.step == 3:
            st.header("4. Adjust RGB Curves")
            sliders = []
            for i, col in enumerate(('R', 'G', 'B')):
                left_val, right_val = st.slider(f'{col} Channel', 0, 255, (st.session_state.results[i]['shift_left_value'] or 0, st.session_state.results[i]['shift_right_value'] or 255))
                sliders.append((left_val, right_val))

            if st.button('Apply Adjustments'):
                st.session_state.adjusted_image = apply_curve_adjustments(st.session_state.spike_removed_image, sliders)
                st.image(st.session_state.adjusted_image, caption='Adjusted Image', use_column_width=True)

        # Step 5: Equalize Histogram
        if st.session_state.step == 4:
            if "adjusted_image" in st.session_state:
                st.header("5. Equalize Histogram")
                if st.button('Equalize Histogram'):
                    st.session_state.equalized_image = equalize_histogram(st.session_state.adjusted_image)
                    st.image(st.session_state.equalized_image, caption='Equalized Image', use_column_width=True)

        # Step 6: Gamma Correction
        if st.session_state.step == 5:
            if "equalized_image" in st.session_state:
                st.header("6. Gamma Correction")
                gamma = st.slider("Gamma Value", 0.1, 3.0, 1.0)
                if st.button('Apply Gamma Correction'):
                    st.session_state.gamma_corrected_image = adjust_gamma(st.session_state.equalized_image, gamma)
                    st.image(st.session_state.gamma_corrected_image, caption='Gamma Corrected Image', use_column_width=True)

        # Step 7: White Balance
        if st.session_state.step == 6:
            if "gamma_corrected_image" in st.session_state:
                st.header("7. White Balance Correction")
                if st.button('Apply White Balance'):
                    st.session_state.white_balanced_image = white_balance(st.session_state.gamma_corrected_image)
                    st.image(st.session_state.white_balanced_image, caption='White Balanced Image', use_column_width=True)

        # Step 8: Apply Extra Enhancements
        if st.session_state.step == 7:
            if "white_balanced_image" in st.session_state:
                st.header("8. Apply Extra Enhancements")
                sharpness_factor = st.slider("Sharpness Factor", 1.0, 2.0, 1.1)
                contrast_factor = st.slider("Contrast Factor", 1.0, 2.0, 1.1)
                if st.button('Apply Extra Enhancements'):
                    enhanced_image = apply_extra_enhancements(st.session_state.white_balanced_image, sharpness_factor, contrast_factor)
                    st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

if __name__ == "__main__":
    main()
