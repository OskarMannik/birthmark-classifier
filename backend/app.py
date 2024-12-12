from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from multimodalmodel import MultimodalModel
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__,
            template_folder=os.path.join(PROJECT_ROOT, 'frontend', 'templates'),
            static_folder=os.path.join(PROJECT_ROOT, 'frontend', 'static'))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'frontend', 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    label_map = {
        0: 'bkl', 1: 'nv', 2: 'df', 3: 'mel',
        4: 'vasc', 5: 'bcc', 6: 'akiec'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tabular_features = 19
    num_classes = len(label_map)
    model = MultimodalModel(num_tabular_features, num_classes).to(device)
    
    model_path = os.path.join(PROJECT_ROOT, 'models', 'final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    return model, label_map, device

def process_inputs(image_path, age, sex, localization):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'localization': [localization]
    })
    
    sex_categories = ['female', 'male']
    localization_categories = [
        'abdomen', 'back', 'chest', 'face', 'lower extremity',
        'neck', 'scalp', 'upper extremity'
    ]
    
    total_categorical_features = len(sex_categories) + len(localization_categories)
    placeholder_features_needed = 19 - (1 + total_categorical_features) 
    
    for i in range(4, 4 + placeholder_features_needed):
        data[f'feature_{i}'] = 0 
    
    numerical_features = ['age'] + [f'feature_{i}' for i in range(4, 4 + placeholder_features_needed)]
    categorical_features = ['sex', 'localization']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(
                categories=[sex_categories, localization_categories],
                sparse_output=False
            ), categorical_features)
        ]
    )
    
    tabular_features_data = preprocessor.fit_transform(data)
    tabular_tensor = torch.tensor(tabular_features_data, dtype=torch.float32)
    
    return image_tensor, tabular_tensor

@app.route('/', methods=['GET'])
def index():
    localizations = ['scalp', 'face', 'neck', 'chest', 'back', 
                    'abdomen', 'upper extremity', 'lower extremity']
    return render_template('index.html', localizations=localizations)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    age = int(request.form['age'])
    sex = request.form['sex']
    localization = request.form['localization']
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    print(image_path)
    
    model, label_map, device = load_model()
    
    image_tensor, tabular_tensor = process_inputs(image_path, age, sex, localization)
    
    image_tensor = image_tensor.to(device)
    tabular_tensor = tabular_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor, tabular_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
    
    prediction = label_map[predicted_class.item()]
    probs_dict = {label_map[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    return jsonify({
        'prediction': prediction,
        'probabilities': probs_dict,
        'image_path': image_path
    })

if __name__ == '__main__':
    app.run(debug=True) 