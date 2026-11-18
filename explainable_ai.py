import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from lime import lime_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.segmentation import mark_boundaries
from datetime import datetime

# === CONFIG ===
MODEL_PATH = 'models/best_model.h5'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load trained model ===
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# === Grad-CAM Function ===
def grad_cam(img_path, model, layer_name='conv5_block3_out'):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = np.dot(conv_outputs[0], weights.numpy())
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = np.uint8(heatmap * 0.4 + orig)

    # Save Grad-CAM result
    filename = os.path.join(RESULTS_DIR, f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(filename, superimposed)
    print(f"💾 Grad-CAM saved: {filename}")

    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM Visualization")
    plt.show()

# === LIME Function ===
def lime_explanation(img_path):
    explainer = lime_image.LimeImageExplainer()
    img = image.load_img(img_path, target_size=(224,224))
    img_array = np.expand_dims(image.img_to_array(img)/255.0, axis=0)

    explanation = explainer.explain_instance(
        img_array[0].astype('double'),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    lime_img = mark_boundaries(temp / 255.0, mask)

    # Save LIME result
    filename = os.path.join(RESULTS_DIR, f"lime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    plt.imsave(filename, lime_img)
    print(f"💾 LIME saved: {filename}")

    plt.imshow(lime_img)
    plt.axis('off')
    plt.title("LIME Visualization")
    plt.show()

# === Run explanation ===
if __name__ == "__main__":
    img_path = input("Enter path of an image to explain (e.g., dataset/train/Brown spot/image1.jpg): ").strip()
    print("\nRunning Grad-CAM...\n")
    grad_cam(img_path, model)
    print("\nRunning LIME...\n")
    lime_explanation(img_path)
    print("\n✅ All results saved in the 'results' folder.")
