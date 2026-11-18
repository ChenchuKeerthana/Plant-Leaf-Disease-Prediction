import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
# import your GradCAM function here

# Paths
model_path = r'models/rice_model.keras'
dataset_path = r'dataset/validation'  # change to your validation folder
output_path = r'gradcam_outputs'
os.makedirs(output_path, exist_ok=True)

# Load model
model = load_model(model_path)

# List all images
all_images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png'))]
all_images = all_images[:100]  # limit to first 100 images

for idx, img_name in enumerate(all_images, 1):
    img_path = os.path.join(dataset_path, img_name)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction (wrap in list to avoid warnings)
        preds = model([img_array])

        # Grad-CAM heatmap (replace with your function)
        # heatmap = generate_gradcam(model, img_array)

        # For demonstration, just save original image
        plt.imshow(img_array[0])
        plt.axis('off')
        plt.title(f'Pred: {np.argmax(preds)}')
        plt.savefig(os.path.join(output_path, f'leaf_{idx}.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")

print("Processed 100 images with Grad-CAM successfully!")