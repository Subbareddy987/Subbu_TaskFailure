from pathlib import Path

import gdown
import joblib
import numpy as np
import streamlit as st


MODEL_FILE = Path("rf_model.pkl")
MODEL_URL = "https://drive.google.com/uc?id=16MtKl_HKbyGMyKtget1wycsHvLEJ_4z6"

LABELS = {
    0: "Enable",
    1: "Evict",
    2: "Lost",
    3: "Finish",
    4: "Kill",
    5: "Fail",
    6: "Queue",
    7: "Schedule",
    8: "Update Pending",
}

EVENT_NOTES = {
    "Enable": "The task is expected to become active or available for execution.",
    "Evict": "The task may be removed because resources are needed elsewhere.",
    "Lost": "The task may be lost due to machine or scheduling issues.",
    "Finish": "The task is expected to complete successfully.",
    "Kill": "The task may be stopped by the system or user.",
    "Fail": "The task has a higher risk of execution failure.",
    "Queue": "The task is expected to wait before execution.",
    "Schedule": "The task is expected to be assigned for execution.",
    "Update Pending": "The task may be waiting for a scheduling or resource update.",
}

SAMPLE_VALUES = {
    "instance_events_type": 0.0,
    "scheduling_class": 1.0,
    "collection_type": 0.0,
    "priority": 3.0,
    "collections_events_type": 0.0,
    "vertical_scaling": 0.0,
    "scheduler": 1.0,
    "start_time": 10.0,
    "end_time": 60.0,
    "assigned_memory": 0.35,
    "page_cache_memory": 0.08,
    "cycles_per_instruction": 1.25,
    "memory_accesses_per_instruction": 0.42,
    "rr_cpu": 0.30,
    "rr_memory": 0.40,
    "au_cpu": 0.22,
    "au_memory": 0.36,
    "mu_cpu": 0.55,
    "mu_memory": 0.62,
}

FIELD_GROUPS = {
    "Task and scheduling details": [
        ("instance_events_type", "Instance Events Type", "Encoded task event type from the dataset."),
        ("scheduling_class", "Scheduling Class", "Encoded scheduling class assigned to the task."),
        ("collection_type", "Collection Type", "Encoded collection or job type."),
        ("priority", "Priority", "Priority level assigned by the scheduler."),
        ("collections_events_type", "Collections Events Type", "Encoded collection event type."),
        ("vertical_scaling", "Vertical Scaling", "Whether vertical scaling is used, encoded numerically."),
        ("scheduler", "Scheduler", "Encoded scheduler value from the dataset."),
    ],
    "Time and memory details": [
        ("start_time", "Start Time", "Task start time value after preprocessing."),
        ("end_time", "End Time", "Task end time value after preprocessing."),
        ("assigned_memory", "Assigned Memory", "Memory assigned to the task."),
        ("page_cache_memory", "Page Cache Memory", "Memory used by page cache."),
    ],
    "CPU and resource usage details": [
        ("cycles_per_instruction", "Cycles Per Instruction", "CPU cycles needed per instruction."),
        (
            "memory_accesses_per_instruction",
            "Memory Accesses Per Instruction",
            "Memory access rate per instruction.",
        ),
        ("rr_cpu", "RR CPU", "Requested CPU resource."),
        ("rr_memory", "RR Memory", "Requested memory resource."),
        ("au_cpu", "AU CPU", "Average used CPU."),
        ("au_memory", "AU Memory", "Average used memory."),
        ("mu_cpu", "MU CPU", "Maximum used CPU."),
        ("mu_memory", "MU Memory", "Maximum used memory."),
    ],
}


st.set_page_config(
    page_title="Cloud Task Failure Prediction",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    if not MODEL_FILE.exists():
        gdown.download(MODEL_URL, str(MODEL_FILE), quiet=True)
    return joblib.load(MODEL_FILE)


def apply_sample_values():
    for key, value in SAMPLE_VALUES.items():
        st.session_state[key] = value


def build_input_array():
    values = []
    for fields in FIELD_GROUPS.values():
        for key, _, _ in fields:
            values.append(st.session_state[key])
    return np.array([values])


st.title("Cloud Task Failure Prediction")
st.caption("Project demo using Google Cluster Trace workload features.")

st.markdown(
    "Enter the task, scheduling, memory, and CPU usage values below. "
    "The trained Random Forest model predicts the likely cloud task event."
)

for fields in FIELD_GROUPS.values():
    for key, _, _ in fields:
        st.session_state.setdefault(key, 0.0)

with st.expander("About this prediction system", expanded=True):
    st.write(
        "This project studies cloud task behavior using selected features from the "
        "Google Cluster Trace Dataset. The model predicts events such as Enable, "
        "Evict, Lost, Finish, Kill, Fail, Queue, Schedule, and Update Pending."
    )

left, right = st.columns([1, 2])
with left:
    st.button("Use sample values", on_click=apply_sample_values)
with right:
    st.info("Use the sample button for a quick demo, or enter your own feature values.")

for group_name, fields in FIELD_GROUPS.items():
    st.subheader(group_name)
    columns = st.columns(2)
    for index, (key, label, help_text) in enumerate(fields):
        with columns[index % 2]:
            st.number_input(
                label,
                step=0.01,
                format="%.4f",
                help=help_text,
                key=key,
            )

if st.session_state["end_time"] < st.session_state["start_time"]:
    st.warning("End Time is lower than Start Time. Please check the time values.")

if st.button("Predict Task Event", type="primary"):
    try:
        model = load_model()
        data = build_input_array()
        prediction = model.predict(data)
        result = LABELS.get(int(prediction[0]), "Unknown")

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data)[0]
            confidence = float(np.max(probabilities)) * 100

        st.success(f"Predicted Event: {result}")

        metric_columns = st.columns(2)
        metric_columns[0].metric("Predicted class", result)
        if confidence is not None:
            metric_columns[1].metric("Model confidence", f"{confidence:.2f}%")
        else:
            metric_columns[1].metric("Model confidence", "Not available")

        st.write(EVENT_NOTES.get(result, "The model returned an unknown event label."))

        if result in {"Fail", "Kill", "Lost", "Evict"}:
            st.error("Risk level: High attention required")
        elif result in {"Queue", "Update Pending"}:
            st.warning("Risk level: Monitor scheduling/resource status")
        else:
            st.info("Risk level: Normal or expected task state")
    except Exception as error:
        st.error("Unable to load the prediction model or complete prediction.")
        st.write("Please check that the Google Drive model link is public and accessible.")
        st.exception(error)
