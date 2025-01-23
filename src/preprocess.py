from PIL import Image
import numpy as np

def preprocess_image(image_file):
    
    img = Image.open(image_file).convert("L")  # Convert to grayscale
    img_resized = img.resize((28, 28))        # Resize to 28x28
    img_array = np.array(img_resized) / 255.0 # Normalize pixel values
    img_inverted = 1 - img_array              # Invert the image
    img_flattened = img_inverted.reshape(1, -1)  # Flatten the image
    return img_flattened
