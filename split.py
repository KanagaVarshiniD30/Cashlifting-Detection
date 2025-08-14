import os
import random
import shutil

base_dir = r"C:/Users/ASUS/Documents/Cashlift/auto_labels"
output_dir = r"C:/Users/ASUS/Documents/Cashlift"

train_img_dir = os.path.join(output_dir, "train/images")
train_lbl_dir = os.path.join(output_dir, "train/labels")
val_img_dir = os.path.join(output_dir, "val/images")
val_lbl_dir = os.path.join(output_dir, "val/labels")

# Create output directories
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)

split_ratio = 0.8
train_count = int(len(images) * split_ratio)

train_files = images[:train_count]
val_files = images[train_count:]

def copy_files(file_list, img_dest, lbl_dest):
    for img_file in file_list:
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.copy(os.path.join(base_dir, img_file), os.path.join(img_dest, img_file))
        shutil.copy(os.path.join(base_dir, lbl_file), os.path.join(lbl_dest, lbl_file))

copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print(f"[INFO] Dataset split complete: {len(train_files)} train, {len(val_files)} val")
