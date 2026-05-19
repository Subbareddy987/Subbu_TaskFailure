import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load GRU Model
gru_model = load_model("gru_model.h5")

# Title
st.title("Cloud Task Failure Prediction")

st.write("Enter Feature Values")

# Input Fields

instance_events_type = st.number_input("Instance Events Type")
scheduling_class = st.number_input("Scheduling Class")
collection_type = st.number_input("Collection Type")
priority = st.number_input("Priority")
collections_events_type = st.number_input("Collections Events Type")
vertical_scaling = st.number_input("Vertical Scaling")
scheduler = st.number_input("Scheduler")
start_time = st.number_input("Start Time")
end_time = st.number_input("End Time")
assigned_memory = st.number_input("Assigned Memory")
page_cache_memory = st.number_input("Page Cache Memory")
cycles_per_instruction = st.number_input("Cycles Per Instruction")
memory_accesses_per_instruction = st.number_input("Memory Accesses Per Instruction")
rr_cpu = st.number_input("RR CPU")
rr_memory = st.number_input("RR Memory")
au_cpu = st.number_input("AU CPU")
au_memory = st.number_input("AU Memory")
mu_cpu = st.number_input("MU CPU")
mu_memory = st.number_input("MU Memory")

# Prediction
if st.button("Predict"):

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

    # Reshape for GRU
    gru_input = data.reshape(
        (data.shape[0], 1, data.shape[1])
    )

    # Prediction
    prediction_probs = gru_model.predict(gru_input)
    st.write(prediction_probs)

    prediction = np.argmax(prediction_probs, axis=1)

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

    st.success(f"Predicted Event: {labels[int(prediction[0])]}")