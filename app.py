import streamlit as st
import numpy as np
import joblib

# Page Config
st.set_page_config(
    page_title="Cloud Task Failure Prediction",
    layout="centered"
)

# Load Models
rf_model = joblib.load("rf_model.pkl")
selector = joblib.load("selector.pkl")

# Title
st.title("Cloud Task Failure Prediction")

st.write("Enter Feature Values")

# Input Fields

instance_events_type = st.number_input("Instance Events Type", value=0.0)
scheduling_class = st.number_input("Scheduling Class", value=0.0)
collection_type = st.number_input("Collection Type", value=0.0)
priority = st.number_input("Priority", value=0.0)
collections_events_type = st.number_input("Collections Events Type", value=0.0)
vertical_scaling = st.number_input("Vertical Scaling", value=0.0)
scheduler = st.number_input("Scheduler", value=0.0)
start_time = st.number_input("Start Time", value=0.0)
end_time = st.number_input("End Time", value=0.0)
assigned_memory = st.number_input("Assigned Memory", value=0.0)
page_cache_memory = st.number_input("Page Cache Memory", value=0.0)
cycles_per_instruction = st.number_input("Cycles Per Instruction", value=0.0)
memory_accesses_per_instruction = st.number_input(
    "Memory Accesses Per Instruction",
    value=0.0
)
rr_cpu = st.number_input("RR CPU", value=0.0)
rr_memory = st.number_input("RR Memory", value=0.0)
au_cpu = st.number_input("AU CPU", value=0.0)
au_memory = st.number_input("AU Memory", value=0.0)
mu_cpu = st.number_input("MU CPU", value=0.0)
mu_memory = st.number_input("MU Memory", value=0.0)

# Predict Button

if st.button("Predict"):

    # Create Input Array
    data = np.array([[
        instance_events_type,
        scheduling_class,
        collection_type,
        priority,
        collections_events_type,
        vertical_scaling,
        scheduler,
        start_time,
        end_time,
        assigned_memory,
        page_cache_memory,
        cycles_per_instruction,
        memory_accesses_per_instruction,
        rr_cpu,
        rr_memory,
        au_cpu,
        au_memory,
        mu_cpu,
        mu_memory
    ]])

    # Feature Selection
    selected_data = selector.transform(data)

    # Random Forest Prediction
    prediction = rf_model.predict(selected_data)

    # Labels
    labels = {
        0: "Enable",
        1: "Evict",
        2: "Lost",
        3: "Finish",
        4: "Kill",
        5: "Fail",
        6: "Queue",
        7: "Schedule",
        8: "Update Pending"
    }

    result = labels.get(int(prediction[0]), "Unknown")

    # Output
    st.success(f"Predicted Event: {result}")
