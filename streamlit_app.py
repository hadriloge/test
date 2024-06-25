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
def analyze_and_plot_histograms(image, corrected=False, sliders=None):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    results = []

    max_hist_value = 0  # Initialize the max histogram value to set y-axis limits

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = hist.flatten()
        max_hist_value = max(max_hist_value, np.max(hist))  # Update the max histogram value

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = hist.flatten()
        ax[i].bar(range(256), hist, color=col)
        ax[i].set_xlim([0, 255])
        ax[i].set_ylim([0, max_hist_value])  # Set y-axis to the max histogram value

        shift_left_value, shift_right_value, significant_spikes = detect_shift(hist)
        spectrum_issue = detect_spectrum_issue(hist)

        results.append({
            'shift_left_value': shift_left_value,
            'shift_right_value': shift_right_value,
            'spectrum_issue': spectrum_issue,
            'significant_spikes': significant_spikes
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
            st.write(f"Significant Spikes: {result['significant_spikes']}")
            st.write("")

    # Add an "About this App" section
    with st.expander("ℹ️ About this App"):
        st.write("""
            This app allows you to analyze and adjust the histograms of an uploaded image. 
            It includes the following steps:
            - **Upload Image**: Choose an image file to upload.
            - **Analysis**: View the RGB histograms and analyze for significant shifts and spikes.
            - **Remove Spikes**: Automatically detect and remove significant spikes in the histogram.
            - **Adjust Significant Values**: Manually adjust the RGB curves using sliders.
            - **Auto-Adjust Brightness**: Automatically adjust the brightness based on the analysis.
            - **Apply Extra Enhancements**: Apply additional enhancements like sharpening and contrast adjustment.
        """)
    
    return results

# Function to detect shifts and significant spikes in the histogram
def detect_shift(hist):
    shift_left_value = None
    shift_right_value = None
    significant_spikes = []

    # Find the first significant left value
    for i in range(len(hist)):
        if hist[i] > 3000:  # Threshold for a significant left shift
            shift_left_value = i
            break

    # Find the first significant right value
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] > 3000:  # Threshold for a significant right shift
            shift_right_value = i
            break

    # Detect significant spikes based on change from one bin to another, within 25 bins from the extremities
    threshold = 0.1 * np.max(hist)  # Threshold for spike detection
    for i in range(1, 25):
        if abs(hist[i] - hist[i - 1]) > threshold and hist[i] > 3000:
            significant_spikes.append((i, hist[i]))
    for i in range(len(hist) - 25, len(hist)):
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

# Function to apply curve adjustments based on slider values
def apply_curve_adjustments(image, sliders):
    adjusted_image = image.copy()
    color = ('b', 'g', 'r')

    for i in range(3):
        left_val, right_val = sliders[i]
        adjusted_image[:, :, i] = np.clip(np.interp(image[:, :, i], [0, left_val, right_val, 255], [0, 0, 255, 255]), 0, 255)

    return adjusted_image

# Function to automatically adjust brightness based on analysis
def auto_adjust_brightness(image, results, underexposure_correction, overexposure_correction):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    for i, result in enumerate(results):
        if result['spectrum_issue'] == "Underexposure":
            v = cv2.add(v, underexposure_correction)
        elif result['spectrum_issue'] == "Overexposure":
            v = cv2.subtract(v, overexposure_correction)

    hsv_image = cv2.merge([h, s, v])
    corrected_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return corrected_image

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

    steps = ["Upload Image", "Analysis", "Remove Spikes", "Adjust Significant Values", "Auto-Adjust Brightness", "Apply Extra Enhancements"]
    
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

        # Step 5: Auto-Adjust Brightness
        if st.session_state.step == 4:
            if "adjusted_image" in st.session_state:
                st.header("5. Auto-Adjust Brightness")
                underexposure_correction = st.slider("Underexposure Correction Value", 0, 100, 20)
                overexposure_correction = st.slider("Overexposure Correction Value", 0, 100, 20)
                if st.button('Auto-Adjust Brightness'):
                    st.session_state.brightness_corrected_image = auto_adjust_brightness(st.session_state.adjusted_image, st.session_state.results, underexposure_correction, overexposure_correction)
                    st.image(st.session_state.brightness_corrected_image, caption='Brightness Corrected Image', use_column_width=True)

        # Step 6: Apply Extra Enhancements
        if st.session_state.step == 5:
            if "brightness_corrected_image" in st.session_state:
                st.header("6. Apply Extra Enhancements")
                sharpness_factor = st.slider("Sharpness Factor", 1.0, 2.0, 1.1)
                contrast_factor = st.slider("Contrast Factor", 1.0, 2.0, 1.1)
                if st.button('Apply Extra Enhancements'):
                    enhanced_image = apply_extra_enhancements(st.session_state.brightness_corrected_image, sharpness_factor, contrast_factor)
                    st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

if __name__ == "__main__":
    main()
