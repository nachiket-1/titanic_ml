# 🚢 Titanic Survival Predictor — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A production-style, end-to-end machine learning project that predicts whether a Titanic passenger would have survived — based on real passenger data. Built with a full sklearn Pipeline, hyperparameter tuning via GridSearchCV, model evaluation, and deployed as an interactive Streamlit web app.

🔗 **Live Demo:** https://titanic-survival-nachiket.streamlit.app
🔗 **Kaggle:** https://www.kaggle.com/code/nachikettalekar/titanic-survival-predictor-end-to-end-ml

---

## 🎯 What This Project Does

You enter details about a Titanic passenger — their age, gender, ticket class, fare, and more — and the model instantly predicts:

- ✅ **Survived** or ❌ **Did Not Survive**
- The **probability/confidence** of the prediction
- A visual **feature importance chart** showing what factors matter most

---

## 🖥️ App Preview

| Passenger Input | Survival Prediction |
|---|---|
| Class, Sex, Age, Fare, Embarked | Live probability with confidence bar |
| Sidebar sliders & dropdowns | Feature importance chart |

---

## 📁 Project Structure

```
titanic_ml/
├── data/
│   └── titanic.csv              # Titanic dataset (from Kaggle)
├── models/
│   ├── titanic_model.pkl        # Saved trained model
│   └── evaluation_plots.png    # Confusion matrix + ROC curve
├── src/
│   ├── preprocess.py            # Feature engineering & sklearn pipeline
│   ├── train.py                 # Model training + hyperparameter tuning
│   └── evaluate.py              # Model evaluation & metrics
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🔄 ML Pipeline Overview

```
Raw Data → Feature Engineering → Preprocessing Pipeline
       → Model Comparison → Hyperparameter Tuning
       → Best Model → Evaluation → Deployment
```

| Stage | Details |
|---|---|
| **Feature Engineering** | Title extraction from names, family size, is-alone flag, age bands, fare bands, cabin flag |
| **Preprocessing** | Median imputation for numerics, mode imputation for categoricals, StandardScaler, OneHotEncoder |
| **Model Comparison** | Logistic Regression vs Random Forest vs Gradient Boosting (5-fold cross-validation) |
| **Hyperparameter Tuning** | GridSearchCV on Random Forest (n_estimators, max_depth, min_samples_split, max_features) |
| **Serialization** | Best model saved with joblib |
| **Deployment** | Interactive Streamlit web app |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~83% |
| Precision | ~81% |
| Recall | ~79% |
| F1 Score | ~80% |
| ROC-AUC | ~87% |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/titanic_ml.git
cd titanic_ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to 👉 [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
- Download `train.csv`, rename it to `titanic.csv`
- Place it inside the `data/` folder

### 4. Train the model
```bash
python src/train.py
```

### 5. Evaluate the model (optional)
```bash
python src/evaluate.py
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📦 Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🧠 Features Used for Prediction

| Feature | Description |
|---|---|
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare paid |
| `Embarked` | Port of embarkation (S/C/Q) |
| `Title` | Extracted from name (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if travelling alone |
| `AgeBand` | Age grouped into 5 bins |
| `FareBand` | Fare grouped into 4 quartiles |
| `HasCabin` | Whether cabin info is known |

---

## 🚀 Deployment

This app is deployed on **Streamlit Cloud** for free.

To deploy your own copy:
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file as `app.py`
5. Click Deploy!

---

## 📚 Dataset

- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Samples:** 891 passengers
- **Features:** 11 original + 6 engineered
- **Target:** Survived (0 = No, 1 = Yes)

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **ML Library:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Saving:** Joblib
- **Web App:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share it.

---

## 🙌 Acknowledgements

- Dataset from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Built as a learning project to demonstrate end-to-end ML development
