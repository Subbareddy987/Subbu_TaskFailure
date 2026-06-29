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
