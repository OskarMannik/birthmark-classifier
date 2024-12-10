# **Birthmark Classifier**

## 👩‍💻 **Authors**
Oskar Männik and Till Bösch

## 🛠️ **Goal of the Project**
Skin cancer affects thousands of individuals each year, with over 1,000 cases reported annually in Estonia. Early detection significantly increases survival rates. Therefore, our goal is to develop a fast and reliable model to assist in diagnosing skin cancer and potentially reduce the number of its cases.

## **Approaches we used**

### Multimodal Skin Cancer Classification Model Overview

Inputs
- **Image**: Shape `(224, 224, 3)` - processed through **ResNet10 layers**.
- **Metadata**: Shape `(8)` - processed through **Dense layers**.

Architecture
1. **Image Branch (ResNet10)**:
   - 4 stages of `BasicBlock`.
   - Global Average Pooling.
   - Output: 512-dimensional feature vector.
2. **Metadata Branch**:
   - Dense layers:
     - `Linear(128) -> ReLU -> Dropout(0.5)`
     - `Linear(64) -> ReLU`.
3. **Fusion Layer**:
   - Combines image and metadata features.
   - Dense layers:
     - `Linear(128) -> ReLU -> Dropout(0.5)`
     - `Linear(num_classes)`.

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

Inputs
- **Image**: Shape `(128, 128, 3)` - processed through **CNN layers**.
- **Metadata**: Shape `(19)` - processed through **DNN layers**.

Architecture
1. **CNN**:
   - 3 Conv2D layers with pooling and global average pooling.
2. **DNN**:
   - 2 dense layers with dropout (`0.5`).

Hyperparameters
- **Optimizer**: `Adam`
- **Loss Function**: `Categorical Crossentropy`
- **Training Settings**: 
  - `40` epochs
  - Batch size: `32`

Results
- **Accuracy**: ~**75.95%**
- **Loss**: ~**0.6313**

Frameworks Used
- **TensorFlow/Keras**
- **Pandas**
- **NumPy**
- **Scikit-learn**

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









