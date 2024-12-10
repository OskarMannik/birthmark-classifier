import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from model_def import MultimodalModel  # Ensure your model definition is imported

# File paths
model_path = "./../models/best_multimodal_model.pth"       # Path to your trained model
metadata_path = "./../data/test_metadata.csv"            # Path to your test metadata
images_folder = "./../data/resized_test_images/"         # Folder containing test images

# Label map for interpretation (use your label mapping)
label_map = {
    0: 'bkl',
    1: 'nv',
    2: 'df',
    3: 'mel',
    4: 'vasc',
    5: 'bcc',
    6: 'akiec'
}

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Verify required columns in metadata
required_columns = ['image_id', 'sex', 'localization', 'age']
if not all(col in metadata.columns for col in required_columns):
    raise ValueError(f"Metadata must contain the following columns: {required_columns}")

# Determine unique values for categorical columns
unique_sex = metadata['sex'].nunique()
unique_localization = metadata['localization'].nunique()

# Dynamically determine the number of additional placeholder features
total_categorical_features = unique_sex + unique_localization
expected_total_features = 19  # Total features used during training
placeholder_features_needed = expected_total_features - (1 + total_categorical_features)  # Subtract 1 for 'age'

# Add only the required placeholder features
for i in range(4, 4 + placeholder_features_needed):
    col_name = f'feature_{i}'
    if col_name not in metadata.columns:
        metadata[col_name] = 0  # Add placeholder values

# Define tabular features
tabular_features = ['age', 'sex', 'localization'] + [f'feature_{i}' for i in range(4, 4 + placeholder_features_needed)]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age'] + [f'feature_{i}' for i in range(4, 4 + placeholder_features_needed)]),
        ('cat', OneHotEncoder(), ['sex', 'localization'])
    ]
)

# Preprocess the tabular data
tabular_features_data = preprocessor.fit_transform(metadata[tabular_features])
tabular_tensor = torch.tensor(tabular_features_data, dtype=torch.float32).to('cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match training dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess images
image_tensors = []
for image_id in metadata['image_id']:
    image_path = os.path.join(images_folder, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensors.append(image_tensor)

# Combine image tensors
image_tensor = torch.cat(image_tensors)

# Load the model
num_tabular_features = tabular_tensor.shape[1]  # Automatically calculate tabular features
num_classes = len(label_map)  # Number of output classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel(num_tabular_features, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Move data to device
image_tensor = image_tensor.to(device)
tabular_tensor = tabular_tensor.to(device)

# Run inference
with torch.no_grad():
    outputs = model(image_tensor, tabular_tensor)
    _, predicted_classes = torch.max(outputs, 1)

# Interpret predictions
predicted_labels = [label_map[class_id.item()] for class_id in predicted_classes]

# Add predictions to metadata
metadata['predicted_label'] = predicted_labels



# Display results
print(metadata[['image_id', 'age', 'sex', 'localization', 'predicted_label']])
