import streamlit as st
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

# -----------------------------
# TITLE
# -----------------------------
st.title("Rice Leaf Disease Detection & XAI")
st.write("Upload a rice leaf image to predict disease and generate Grad-CAM and LIME explanations.")

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = os.path.join("models", "rice_model.keras")

try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except:
    st.error("Model not found. Put rice_model.keras inside models folder.")
    st.stop()

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
"Brown Spot",
"Leaf Blast",
"Leaf Scald",
"Leaf Smut",
"Bacterial Leaf Blight",
"Healthy"
]

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=400)

    # -----------------------------
    # PREPROCESS IMAGE
    # -----------------------------
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    img_tensor = tf.convert_to_tensor(img_array)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    preds = model.predict(img_tensor)[0]

    predicted_class = class_names[np.argmax(preds)]

    st.subheader("Prediction Result")
    st.write("Predicted Disease:", predicted_class)

    # -----------------------------
    # CONFIDENCE GRAPH
    # -----------------------------
    st.subheader("Prediction Confidence")

    num_classes = len(preds)

    if len(class_names) != num_classes:
        class_names = [f"Class {i+1}" for i in range(num_classes)]

    fig, ax = plt.subplots()
    ax.bar(class_names, preds)
    ax.set_ylabel("Probability")
    ax.set_title("Disease Confidence Score")

    st.pyplot(fig)

    # -----------------------------
    # GRAD CAM
    # -----------------------------
    def make_gradcam_heatmap(img_tensor, model):

        last_conv_layer = None

        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(img_tensor)

            class_idx = tf.argmax(predictions[0])

            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()


    def overlay_heatmap(img, heatmap):

        heatmap = cv2.resize(heatmap,(img.width,img.height))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_np = np.array(img)

        superimposed = cv2.addWeighted(img_np,0.6,heatmap,0.4,0)

        return superimposed


    # -----------------------------
    # SHOW GRAD-CAM
    # -----------------------------
    try:

        st.subheader("Grad-CAM Explanation")

        heatmap = make_gradcam_heatmap(img_tensor, model)

        cam = overlay_heatmap(img, heatmap)

        st.image(cam, width=400)

        # SAVE IMAGE
        gradcam_path = "outputs/gradcam_result.jpg"
        cv2.imwrite(gradcam_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

        st.success("Grad-CAM image saved in outputs folder")

    except Exception as e:

        st.error(f"Grad-CAM error: {e}")


    # -----------------------------
    # LIME EXPLANATION
    # -----------------------------
    try:

        st.subheader("LIME Explanation")

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            np.array(img_resized).astype("double"),
            classifier_fn=lambda x: model.predict(x/255.0),
            top_labels=1,
            hide_color=0,
            num_samples=500
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        lime_img = mark_boundaries(temp/255.0, mask)

        st.image(lime_img, width=400)

        # SAVE LIME IMAGE
        lime_path = "outputs/lime_result.jpg"
        cv2.imwrite(lime_path, (lime_img*255).astype("uint8"))

        st.success("LIME image saved in outputs folder")

    except Exception as e:

        st.error(f"LIME error: {e}")