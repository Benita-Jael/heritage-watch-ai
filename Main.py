import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF
from tensorflow.keras.preprocessing import image as keras_image
import os

# ✅ Set page config FIRST before any other Streamlit commands
st.set_page_config(page_title="Heritage Damage Classifier", layout="centered")

# Load models
@st.cache_resource
def load_models():
    damage_model = tf.keras.models.load_model("/content/drive/MyDrive/models/final_damage_detection_model.h5")
    enhancer_model = tf.keras.models.load_model("/content/drive/MyDrive/models/zero_dce_tf_model.h5")
    defect_model = tf.keras.models.load_model("/content/drive/MyDrive/models/final_defect_classification_model.h5")
    return damage_model, enhancer_model, defect_model
    

damage_model, enhancer_model, defect_model = load_models()

# Class names
class_names = ['IMPACT DETECTED - DAMAGED', 'INTACT - NO DAMAGE']
defect_classes = ['Missing_cracks', 'Stains_yellowing_plantmoss']

# Zero-DCE post-processing
def post_process(image, output):
    r = [output[:, :, :, i:i+3] for i in range(0, 24, 3)]
    x = image
    for ri in r:
        x = x + ri * (tf.square(x) - x)
    return x

def enhance_low_light(image_pil):
    image = tf.keras.preprocessing.image.img_to_array(image_pil).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = enhancer_model(image)
    enhanced = post_process(image, output)
    enhanced = tf.clip_by_value(enhanced, 0.0, 1.0)
    enhanced = tf.cast(enhanced[0] * 255.0, dtype=tf.uint8)
    return Image.fromarray(enhanced.numpy())

def is_dark_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return brightness < 100, brightness

# Grad-CAM with correct layer for Keras model
def generate_gradcam_keras(model, img_array, class_index, last_conv_layer_name):
    base_model = model.get_layer("inception_v3")

    grad_model = tf.keras.models.Model(
        [base_model.input],
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img = np.squeeze(img_array) * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite("gradcam_output.jpg", superimposed_img)

# Streamlit UI
st.title("🔔 Heritage Structure Damage Detection")
st.write("Upload an image to classify it as **Damaged** or **Intact**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    is_dark, brightness = is_dark_image(image)
    st.write(f"📊 Brightness Score: {brightness:.2f}")

    if is_dark:
        st.warning("🌑 Low-light image detected. Enhancing using Zero-DCE...")
        image = enhance_low_light(image)
        st.image(image, caption="🔧 Enhanced Image", use_column_width=True)
    else:
        st.success("✅ Image brightness is sufficient.")

    img_array = np.array(image)
    resized_img = cv2.resize(img_array, (224, 224))
    preprocessed = resized_img.astype("float32") / 255.0
    preprocessed = np.expand_dims(preprocessed, axis=0)

    prediction = damage_model.predict(preprocessed)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    st.markdown(f"### 🔍 Prediction: **{class_names[predicted_class]}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    cv2.imwrite("enhanced_image.jpg", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    if predicted_class == 0:
        st.subheader("🔎 Defect Classification")
        defect_pred = defect_model.predict(preprocessed)
        defect_class = np.argmax(defect_pred[0])
        defect_label = defect_classes[defect_class]
        st.markdown(f"**Defect Type:** {defect_label}")

        st.subheader("🔥 Grad-CAM Heatmap")
        generate_gradcam_keras(defect_model, preprocessed, defect_class, last_conv_layer_name="conv2d_93")
        st.image("gradcam_output.jpg", caption="Grad-CAM Heatmap", use_column_width=True)

        st.subheader("📄 Download Report")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="HERITAGE WATCH AI REPORT", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Brightness Score: {brightness:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Damage Prediction: {class_names[predicted_class]}", ln=True)
        pdf.cell(200, 10, txt=f"Defect Type: {defect_label}", ln=True)
        pdf.image("enhanced_image.jpg", x=10, y=50, w=90)
        pdf.image("gradcam_output.jpg", x=110, y=50, w=90)
        report_path = "report.pdf"
        pdf.output(report_path)

        with open(report_path, "rb") as f:
            st.download_button("🗕️ Download Report PDF", f, file_name="heritage_report.pdf")
