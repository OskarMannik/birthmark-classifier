import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch.nn.functional as F
import pandas as pd
from multimodalmodel import MultimodalModel  

model_path = "./../models/final_model.pth"      
metadata_path = "./../data/test_metadata.csv"           
images_folder = "./../data/resized_test_images/"        

label_map = {
    0: 'bkl',
    1: 'nv',
    2: 'df',
    3: 'mel',
    4: 'vasc',
    5: 'bcc',
    6: 'akiec'
}

metadata = pd.read_csv(metadata_path)

required_columns = ['image_id', 'sex', 'localization', 'age']
if not all(col in metadata.columns for col in required_columns):
    raise ValueError(f"Metadata must contain the following columns: {required_columns}")

unique_sex = metadata['sex'].nunique()
unique_localization = metadata['localization'].nunique()

total_categorical_features = unique_sex + unique_localization
expected_total_features = 19  
placeholder_features_needed = expected_total_features - (1 + total_categorical_features) 

for i in range(4, 4 + placeholder_features_needed):
    col_name = f'feature_{i}'
    if col_name not in metadata.columns:
        metadata[col_name] = 0 

tabular_features = ['age', 'sex', 'localization'] + [f'feature_{i}' for i in range(4, 4 + placeholder_features_needed)]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age'] + [f'feature_{i}' for i in range(4, 4 + placeholder_features_needed)]),
        ('cat', OneHotEncoder(), ['sex', 'localization'])
    ]
)

tabular_features_data = preprocessor.fit_transform(metadata[tabular_features])
tabular_tensor = torch.tensor(tabular_features_data, dtype=torch.float32).to('cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensors = []
for image_id in metadata['image_id']:
    image_path = os.path.join(images_folder, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) 
    image_tensors.append(image_tensor)

image_tensor = torch.cat(image_tensors)

num_tabular_features = tabular_tensor.shape[1] 
num_classes = len(label_map)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel(num_tabular_features, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

image_tensor = image_tensor.to(device)
tabular_tensor = tabular_tensor.to(device)

with torch.no_grad():
    outputs = model(image_tensor, tabular_tensor)
    _, predicted_classes = torch.max(outputs, 1)

predicted_labels = [label_map[class_id.item()] for class_id in predicted_classes]

metadata['predicted_label'] = predicted_labels


print(metadata[['image_id', 'age', 'sex', 'localization', 'predicted_label']])

with torch.no_grad():
    outputs = model(image_tensor, tabular_tensor)
    probabilities = F.softmax(outputs, dim=1)
    _, predicted_classes = torch.max(outputs, 1)
    print(probabilities)

'''# Convert probabilities to a more readable format
probability_list = probabilities.cpu().numpy().tolist()
probability_columns = [f'prob_{label_map[i]}' for i in range(num_classes)]

# Add probabilities and predictions to metadata
probability_df = pd.DataFrame(probability_list, columns=probability_columns)
metadata = pd.concat([metadata.reset_index(drop=True), probability_df], axis=1)

predicted_labels = [label_map[class_id.item()] for class_id in predicted_classes]
metadata['predicted_label'] = predicted_labels

# Display results
columns_to_display = ['image_id', 'age', 'sex', 'localization', 'predicted_label'] + probability_columns
print(metadata[columns_to_display])'''
