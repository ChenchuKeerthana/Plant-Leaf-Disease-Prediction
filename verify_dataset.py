import os

DATASET_DIR = "dataset/train"

if not os.path.exists(DATASET_DIR):
    print("❌ Folder not found:", DATASET_DIR)
else:
    print(f"✅ Found folder: {DATASET_DIR}\n")
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print("Detected Classes:")
    total_images = 0
    for cls in classes:
        path = os.path.join(DATASET_DIR, cls)
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_images += count
        print(f" - {cls}: {count} images")
    print(f"\n📊 Total images: {total_images}")
