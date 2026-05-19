# Subbu_TaskFailure

A Deep Learning based cloud task failure prediction system using GRU and Machine Learning classifiers on the Google Cluster Trace Dataset.

---

##  Project Overview

This project focuses on the early prediction of task failures in cloud computing environments using a hybrid Deep Learning + Machine Learning framework.

The system uses:

- SelectKBest for feature selection
- GRU (Gated Recurrent Unit) for feature extraction
- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbor (KNN)

The model predicts different task states/events in Google cloud clusters such as:

- Enable
- Evict
- Lost
- Finish
- Kill
- Fail
- Queue
- Schedule
- Update Pending

The framework helps reduce:

- Resource wastage
- SLA violations
- Task execution failures
- Cloud reliability issues

---

# 🧠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

#  Dataset

Dataset Used:

- Google Cluster Trace Dataset

Dataset contains:

- CPU usage
- Memory usage
- Scheduling information
- Task events
- Resource requests
- Task execution states

---

#  Project Workflow

## 1. Data Preprocessing
- Remove unnecessary columns
- Handle missing values
- Normalize data using MinMaxScaler

## 2. Feature Selection
- SelectKBest algorithm used
- Top important features selected

## 3. Feature Extraction
- GRU network extracts temporal patterns
- Learns sequential dependencies

## 4. Classification

Machine Learning classifiers used:

- Random Forest
- SVM
- KNN

## 5. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- RMSE

---

# 📊 Model Performance

- GRU achieved around 97.7% accuracy
- GRU + Random Forest gave best performance
- Average AUC per class > 0.98

---

#  Files Included

- Final_min.ipynb → Main implementation notebook
- ABC.csv → Dataset file

---

#  Future Improvements

- Deploy as a web application
- Real-time cloud monitoring
- Integrate live cloud workloads
- Add dashboard visualizations
- Improve scalability

---

#  Author

Subba Reddy

GitHub Repository:
https://github.com/Subbareddy987/Subbu_TaskFailure
