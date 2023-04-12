import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from utils import hugprocess_img


file_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(file_root, "models", "img", "evgeni-evgeniev-ggVH1hoQAac-unsplash.jpg")


hugprocess_img = hugprocess_img(image_path, augment=True)
print("Preprocessed image shape: ", hugprocess_img.shape)