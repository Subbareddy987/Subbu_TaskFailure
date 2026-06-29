# Cloud Task Failure Prediction

A final year project for predicting cloud task events and possible task failures using machine learning on the Google Cluster Trace Dataset.

Live demo: https://subbu-sps-taskfailure-xyz.streamlit.app/

GitHub repository: https://github.com/Subbareddy987/Subbu_TaskFailure

## Project Objective

Cloud computing platforms run a large number of tasks at the same time. Some tasks may fail, get killed, wait in a queue, or be evicted because of resource and scheduling issues. This project predicts the likely task event from workload features such as CPU usage, memory usage, scheduling details, priority, and task timing.

The aim is to support early identification of risky task behavior so that cloud resources can be used more efficiently.

## Key Features

- Predicts cloud task events from user-entered feature values
- Uses selected workload and resource usage features
- Provides a deployed Streamlit web application
- Shows the predicted task event and a short interpretation
- Supports quick testing with sample input values
- Supports CSV upload for batch prediction

## Predicted Event Classes

The model predicts one of the following task states:

- Enable
- Evict
- Lost
- Finish
- Kill
- Fail
- Queue
- Schedule
- Update Pending

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Gdown
- Jupyter Notebook
- TensorFlow / Keras
- Matplotlib
- Seaborn

## Dataset

Dataset used: Google Cluster Trace Dataset

The dataset contains cloud workload information such as:

- CPU usage
- Memory usage
- Scheduling information
- Task events
- Resource requests
- Task execution states

## Project Workflow

### 1. Data Preprocessing

- Removed unnecessary columns
- Handled missing values
- Normalized numerical features
- Prepared task event labels for classification

### 2. Feature Selection

- Used SelectKBest to identify important features
- Reduced the feature set to the most useful task and resource attributes

### 3. Feature Extraction

- Used GRU-based deep learning concepts to learn workload behavior and temporal patterns

### 4. Classification

Machine learning classifiers used in the project include:

- Random Forest
- Support Vector Machine
- K-Nearest Neighbor

The deployed app uses the trained Random Forest model for prediction.

### 5. Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC curve
- RMSE

## Model Performance

- GRU achieved around 97.7% accuracy
- GRU + Random Forest gave the best performance
- Average AUC per class was greater than 0.98

## Application Overview

The Streamlit app accepts feature values related to:

- Task and scheduling details
- Time and memory details
- CPU and resource usage details

After clicking the prediction button, the app displays:

- Predicted event class
- Confidence score when available
- Short explanation of the predicted event
- Basic risk level interpretation

The app also supports CSV upload for batch prediction. This allows multiple task records to be predicted at once, which is closer to a real cloud monitoring scenario.

## How To Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Files Included

- `app.py` - Streamlit web application
- `Final_min.ipynb` - Main model development notebook
- `ABC.csv` - Dataset file used for the project
- `selector.pkl` - Saved feature selector
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Deployment Note

The deployed app downloads the trained model file from Google Drive when `rf_model.pkl` is not already present. This keeps the repository lighter, but the Google Drive link must remain accessible for deployment to work correctly.

## Future Improvements

- Add live cloud workload monitoring
- Add dashboard visualizations for CPU and memory patterns
- Improve preprocessing pipeline documentation
- Add model comparison charts inside the app
- Add screenshots and architecture diagrams to the README

## Author

Subba Reddy
