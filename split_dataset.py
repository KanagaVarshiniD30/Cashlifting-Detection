import os
import random
import shutil

# Base dataset path
dataset_path = r"C:/Users/ASUS/Documents/Cashlift/processed_frames"
normal_path = os.path.join(dataset_path, "processed_normal")
cashlifting_path = os.path.join(dataset_path, "processed_cashlift")

# Temporary merged image/label folders
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)


def move_files(src_folder):
    """Copy images and labels from a source folder into merged paths."""
    for file in os.listdir(src_folder):
        src_file = os.path.join(src_folder, file)
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy(src_file, os.path.join(images_path, file))
        elif file.lower().endswith(".txt"):
            shutil.copy(src_file, os.path.join(labels_path, file))


# Merge normal and cashlifting data
move_files(normal_path)
move_files(cashlifting_path)

# Create train/val folder structure
train_img_dir = os.path.join(dataset_path, "train", "images")
train_lbl_dir = os.path.join(dataset_path, "train", "labels")
val_img_dir = os.path.join(dataset_path, "val", "images")
val_lbl_dir = os.path.join(dataset_path, "val", "labels")

for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(path, exist_ok=True)

# Split ratio
split_ratio = 0.8
image_files = [f for f in os.listdir(images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(image_files)
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]


def copy_files(file_list, img_dest, lbl_dest):
    """Copy images and corresponding labels; create empty label if missing."""
    for img_file in file_list:
        lbl_file = os.path.splitext(img_file)[0] + ".txt"

        # Copy image
        shutil.copy(os.path.join(images_path, img_file), os.path.join(img_dest, img_file))

        # Copy or create label
        label_src = os.path.join(labels_path, lbl_file)
        label_dst = os.path.join(lbl_dest, lbl_file)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            with open(label_dst, "w") as f:
                pass  # empty file
            print(f"[INFO] Created empty label for: {img_file}")


# Copy train and val data
copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print(f"[INFO] Dataset prepared and split successfully!")
print(f"Train images: {len(train_files)}, Val images: {len(val_files)}")
