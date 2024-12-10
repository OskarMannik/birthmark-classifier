# **Birthmark Classifier**

## 👩‍💻 **Authors**
Oskar Männik and Till Bösch

## 🛠️ **Goal of the Project**
Skin cancer affects thousands of individuals each year, with over 1,000 cases reported annually in Estonia. Early detection significantly increases survival rates. Therefore, our goal is to develop a fast and reliable model to assist in diagnosing skin cancer and potentially reduce the number of its cases.

## **Approaches we used**

### Multimodal Skin Cancer Classification Model Overview (final result)

Inputs
- **Image**: `(224, 224, 3)` - processed through **ResNet10**.
- **Metadata**: `(8)` - processed through **Dense layers**.

Architecture
- **Image Branch**: ResNet10 with 512-dimensional output.
- **Metadata Branch**: Dense layers with ReLU and Dropout.
- **Fusion Layer**: Combines features from both branches for classification.

Hyperparameters
- **Optimizer**: `AdamW`
- **Loss Function**: `Focal Loss` (`alpha=1, gamma=2`)
- **Learning Rate Scheduler**: `StepLR` (reduce LR by 10% every 5 epochs)
- **Training Settings**: 
  - `20` epochs
  - Batch size: `32`

Results
- **Test Accuracy**: ~**75.99%**
- **Test Loss**: ~**0.1563**

Frameworks Used
- **PyTorch/Torchvision**
- **Scikit-learn**
- **Pandas**
- **PIL**

---

### Basic CNN Model Overview

- **Inputs**: Image `(128, 128, 3)` via CNN, Metadata via DNN.  
- **Architecture**: 3 Conv2D + pooling (CNN), 2 dense layers + dropout (DNN).  
- **Hyperparameters**: Adam, Categorical Crossentropy, 40 epochs, batch size 32.  
- **Results**: Accuracy ~75.95%, Loss ~0.6313.  
- **Frameworks**: TensorFlow/Keras, Pandas, NumPy, Scikit-learn.  

---

### ResNet15 Model Overview

- **Inputs**: Image `(224, 224, 3)` via ResNet18, Metadata via DNN.  
- **Architecture**: Pretrained ResNet18 (CNN), 2 dense layers (DNN), fusion layer for combined features.  
- **Hyperparameters**: Adam, CrossEntropyLoss, 10 epochs, batch size 32.  
- **Results**: Accuracy ~79.23%, Loss ~0.5984.  
- **Frameworks**: PyTorch, Torchvision, scikit-learn.
   
---

### ResNet50 Model Overview

- **Inputs**: Image `(124, 124, 3)` via ResNet50, Metadata `(5)` via DNN.  
- **Architecture**: Pretrained ResNet50, 2 dense layers (DNN), fusion layer for combined features.
- **Hyperparameters**: Adam, Categorical Crossentropy, 5 epochs, batch size 32.  
- **Results**: Accuracy ~72.09%, Loss ~0.8577.  
- **Frameworks**: TensorFlow/Keras, Pandas, NumPy, Scikit-learn.  

---
### EfficientNetB0 Model Overview

- **Inputs**: Image `(224, 224, 3)` via EfficientNetB0, Metadata via DNN.  
- **Architecture**: Pretrained EfficientNetB0, Flatten, 2 dense layers (DNN), 3 fusion layers for combined features.  
- **Hyperparameters**: Adam (1e-4), Sparse Categorical Crossentropy, 20 epochs, batch size 16, class weights applied.  
- **Results**: Accuracy ~73.16%, Loss ~1.6875.  
- **Frameworks**: TensorFlow/Keras, Pandas, NumPy, Matplotlib.  

---
### CNN Model with Undersampling Overview

- **Inputs**: Image `(228, 228, 3)` via CNN, Metadata via DNN.  
- **Architecture**: 3 Conv2D layers + GlobalPooling (CNN), 2 Dense layers (DNN), 2 fusion layers for combined features.  
- **Hyperparameters**: Adam, Categorical Crossentropy, 20 epochs, batch size 32.  
- **Results**: Accuracy ~58.44%, Loss ~1.0152.  
- **Frameworks**: TensorFlow/Keras, Pandas, NumPy, Matplotlib.  

---
### CNN Model with SMOTE and Data Augmentation Overview

- **Inputs**: Image `(128, 128, 3)` via CNN, Metadata via DNN.  
- **Architecture**: 3 Conv2D layers + GlobalPooling (CNN), 2 Dense layers (DNN), 2 fusion layers for combined features.  
- **Hyperparameters**: Adam, Categorical Crossentropy, 20 epochs, batch size 32.  
- **Techniques**: SMOTE for oversampling, ImageDataGenerator for augmentation.  
- **Results**: Accuracy ~78.00%, Loss ~0.5915.  
- **Frameworks**: TensorFlow/Keras, Pandas, NumPy, Matplotlib.  

---
Resnet, Efficientnet,CNN from scratch
Image preprocessing (size)
Different optimizers and parameters
Weights adjustment
Tensorflow, pytorch
Undersampling, oversampling
Scheduler
Focal Loss



## 📂 **Guide to the Contents of the Repository**
```markdown
project/
├── 🗂️ backend/              # Backend code for app logic
├── 🗂️ data/                 # Some files of the dataset
├── 🗂️ frontend/             # Code for the user interface
├── 🗂️ models/               # Our own trained and tested models. The main one is final_model.pth
├── 🗂️ notebooks/            # Python notebooks that we used to train and test our models
├── 🗂️ _pycache_/            
├── 📄 C1_report.pdf         # project report
├── 📄 README.md             
└── 📄 requirements.txt      # Python dependencies for the project
```

## 🚀 **Try it Yourself!**

### 1. Clone the repository

```bash
git clone https://github.com/OskarMannik/birthmark-classifier.git
```
```bash
cd birthmark-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the notebook

Run the ```final_notebook.ipynb``` to download the dataset and to train model.

### 4. Try it with the UI

Go to backend folder and edit ```app.py```. Replace ```final_model.pth``` with your previously trained and saved model. To test your model, run ```app.py``` locally with: 
```bash 
python3 app.py
```

### 5. Enjoy!









