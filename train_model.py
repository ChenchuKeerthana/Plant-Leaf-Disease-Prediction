import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==== Paths ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')       # path to train folder
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')  # path to validation folder
MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'models', 'rice_model.keras')

# ==== Parameters ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# ==== Data Generators ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15,
    shear_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Only use a subset of images (500 for training + 200 for validation)
# You can create a temp folder with the first 500 images per class for speed
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==== Model ====
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base layers for initial training (optional)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# ==== Training ====
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Optionally, unfreeze some top layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile and fine-tune
model.compile(optimizer=Adam(LEARNING_RATE/10), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# ==== Save Model ====
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")