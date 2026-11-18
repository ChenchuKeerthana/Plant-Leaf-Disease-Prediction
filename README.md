# Plant Leaf Disease Detection using Explainable AI (XAI)

## Overview

This project focuses on detecting diseases in rice plant leaves using deep learning techniques and providing explanations for the model's predictions using Explainable AI (XAI) methods.

The system uses a Convolutional Neural Network (CNN) model to classify rice leaf diseases and employs Grad-CAM and LIME techniques to visualize and explain the decision-making process of the model.

This helps improve transparency and trust in AI-based agricultural systems.

## Features

* Deep learning based rice leaf disease classification
* Explainable AI visualization
* Grad-CAM heatmap generation
* LIME-based model interpretation
* Interactive prediction using a Python application

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* LIME
* Grad-CAM

## Project Structure

plant-leaf-disease-detection
│
├── dataset/                # Rice leaf disease dataset
├── src/                    # Model training and processing code
├── models/                 # Trained deep learning models
├── outputs/                # Prediction results
├── gradcam_outputs/        # GradCAM visualizations
├── lime_outputs/           # LIME explanation outputs
├── xai_results/            # Combined XAI results
├── app.py                  # Application for prediction
├── rrr.py                  # Supporting scripts

## Explainable AI Techniques

### Grad-CAM

Grad-CAM generates heatmaps that highlight the important regions of the leaf image which influence the model's prediction.

### LIME

LIME explains individual predictions by approximating the model locally and highlighting influential features.

## Applications

* Smart agriculture
* Early plant disease detection
* Crop health monitoring
* AI-assisted farming systems

## Future Improvements

* Integration with mobile applications
* Real-time disease detection using camera input
* Deployment using cloud platforms
* Expansion to multiple crop diseases

## Author

Balumuru Chaithanya
B.Tech Computer Science Engineering
Interest Areas: Artificial Intelligence, Machine Learning, Explainable AI

## License

This project is intended for research and educational purposes.
