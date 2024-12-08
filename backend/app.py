from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from model_def import MultimodalModel
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './frontend/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_model():
    # Label map for interpretation
    label_map = {
        0: 'bkl', 1: 'nv', 2: 'df', 3: 'mel',
        4: 'vasc', 5: 'bcc', 6: 'akiec'
    }
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tabular_features = 19  # Your total features
    num_classes = len(label_map)
    model = MultimodalModel(num_tabular_features, num_classes).to(device)
    model.load_state_dict(torch.load("final_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    return model, label_map, device

def process_inputs(image_path, age, sex, localization):
    # Image processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Create DataFrame for tabular data
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'localization': [localization]
    })
    
    # Define the exact categories for each categorical feature
    sex_categories = ['female', 'male']  # Make sure these match your training data
    localization_categories = [
        'abdomen', 'back', 'chest', 'face', 'lower extremity',
        'neck', 'scalp', 'upper extremity'
    ]
    
    # Add placeholder features
    for i in range(4, 12):  # Adding 8 placeholder features
        data[f'feature_{i}'] = 0
    
    # Define features order
    numerical_features = ['age'] + [f'feature_{i}' for i in range(4, 12)]
    categorical_features = ['sex', 'localization']
    
    # Create preprocessor with fixed categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(
                categories=[sex_categories, localization_categories],
                sparse_output=False
            ), categorical_features)
        ]
    )
    
    # Process tabular data
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
    
    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    
    # Load model
    model, label_map, device = load_model()
    
    # Process inputs
    image_tensor, tabular_tensor = process_inputs(image_path, age, sex, localization)
    
    # Move to device
    image_tensor = image_tensor.to(device)
    tabular_tensor = tabular_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor, tabular_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
    
    # Get prediction and probabilities
    prediction = label_map[predicted_class.item()]
    probs_dict = {label_map[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    return jsonify({
        'prediction': prediction,
        'probabilities': probs_dict,
        'image_path': image_path
    })

if __name__ == '__main__':
    app.run(debug=True) 