import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# File paths
model_path = "best_multimodal_model.pth"
metadata_path = "HAM10000_metadata.csv"
image_paths = [
    "resized_test_images/ISIC_0024321.jpg",
    "resized_test_images/ISIC_0024311.jpg",
    "resized_test_images/ISIC_0024317.jpg",
]

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Define transformation for the images (same as training/validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.eval()

# Prepare the tabular data for the images
# Assuming metadata has a column `image_id` that matches the image filenames and
# tabular features needed for the model are all numeric columns
tabular_columns = metadata.select_dtypes(include='number').columns
tabular_data = metadata[metadata['image_id'].isin([os.path.basename(p).split('.')[0] for p in image_paths])]
tabular_features = tabular_data[tabular_columns].values

# Process each image and run predictions
results = []

for idx, image_path in enumerate(image_paths):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get the corresponding tabular data
    tabular_tensor = torch.tensor(tabular_features[idx], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output = model(image_tensor, tabular_tensor)
        _, predicted_class = torch.max(output, 1)
        results.append((os.path.basename(image_path), predicted_class.item()))

results



for idx, (image_name, predicted_class) in enumerate(results):
    image = Image.open(image_paths[idx])
    plt.imshow(image)
    plt.title(f"Image: {image_name}\nPredicted Class: {predicted_class}")
    plt.axis("off")
    plt.show()
