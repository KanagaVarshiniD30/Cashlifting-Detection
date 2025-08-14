import cv2
import os
from pathlib import Path

def preprocess_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_file in Path(input_dir).glob("*.jpg"):
        img = cv2.imread(str(img_file))

        # Resize
        img = cv2.resize(img, (640, 640))

        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Brightness/contrast normalization
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

        # Blur detection (skip very blurry)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:  # low score = too blurry
            continue

        # Save processed image
        cv2.imwrite(str(Path(output_dir) / img_file.name), img)

# Run for both datasets
preprocess_images("C:/Users/ASUS/Documents/Cashlift/data/images/normal", "C:/Users/ASUS/Documents/Cashlift/processed_frames/processed_normal")
preprocess_images("C:/Users/ASUS/Documents/Cashlift/data/images/cashlifting", "C:/Users/ASUS/Documents/Cashlift/processed_frames/processed_cashlift")
