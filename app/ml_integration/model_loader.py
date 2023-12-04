import joblib
import numpy as np


def classify_survived(list_of_features: list) -> str:

    model = joblib.load('ml_integration/classification_model.joblib')
    test_data = np.array(list_of_features).reshape(1, -1)
    prediction = model.predict(test_data)
    return ([prediction[0]])

