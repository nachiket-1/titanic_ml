# 🚢 Titanic Survival Predictor — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)

A production-style end-to-end ML project that predicts whether a Titanic passenger would have survived.

🔗 **Live Demo:** [titanic-survival-nachiket.streamlit.app](https://titanic-survival-nachiket.streamlit.app)

---

## 🎯 What This Project Does
Enter passenger details and the model predicts:
- ✅ Survived or ❌ Did Not Survive
- The probability/confidence of the prediction
- A visual feature importance chart

---

## 📁 Project Structure
```
titanic_ml/
├── data/
│   └── titanic.csv
├── models/
│   └── titanic_model.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── app.py
└── requirements.txt
```

---

## 🔄 ML Pipeline
| Stage | Details |
|---|---|
| Feature Engineering | Title extraction, family size, age bands, fare bands |
| Preprocessing | Imputation, StandardScaler, OneHotEncoder |
| Model Comparison | Logistic Regression vs Random Forest vs Gradient Boosting |
| Hyperparameter Tuning | GridSearchCV with 5-fold cross-validation |
| Deployment | Streamlit web app |

---

## 📊 Model Performance
| Metric | Score |
|---|---|
| Accuracy | ~83% |
| F1 Score | ~80% |
| ROC-AUC | ~87% |

---

## ⚙️ Run Locally
```bash
git clone https://github.com/nachiket-1/titanic_ml.git
cd titanic_ml
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

---

## 🛠️ Tech Stack
- Python, Scikit-learn, Pandas, NumPy
- Streamlit, Matplotlib, Seaborn, Joblib

---

## 📚 Dataset
[Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)