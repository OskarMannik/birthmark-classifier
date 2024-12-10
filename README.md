# **Birthmark Classifier**

## 👩‍💻 **Authors**
Oskar Männik and Till Bösch

## 🛠️ **Goal of the Project**
Skin cancer affects thousands of individuals each year, with over 1,000 cases reported annually in Estonia. Early detection significantly increases survival rates. Therefore, our goal is to develop a fast and reliable model to assist in diagnosing skin cancer and potentially reduce the number of its cases.

## **Approaches we used**

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









