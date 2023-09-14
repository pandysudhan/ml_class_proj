import joblib
import numpy as np


def find_type_iris(list_of_features: list) -> str:

    types_of_flower = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    model = joblib.load('app/ml_integration/ml_model_test.joblib')
    test_data = np.array(list_of_features).reshape(1, -1)
    prediction = model.predict(test_data)
    return (types_of_flower[prediction[0]])

print(find_type_iris([4.7, 3.2, 1.2, 0.2]))