import os
from transformers import pipeline

ACTIVITY_CLASSIFIER_PATH = "final_activity_classifier_deberta_PMo"

def load_classifier():
    if os.path.exists(ACTIVITY_CLASSIFIER_PATH):
        try:
            print("--- Loading local Activity Granularity Classifier model... ---")
            return pipeline("text-classification", model=ACTIVITY_CLASSIFIER_PATH, tokenizer=ACTIVITY_CLASSIFIER_PATH)
        except Exception as e:
            print(f"!!! WARNING: Could not load local activity classifier: {e} !!!")
    return None

activity_classifier_pipeline = load_classifier()