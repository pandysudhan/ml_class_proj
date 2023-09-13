

# stand alone code
import numpy as np
import pandas as pd
from sklearn import (datasets, metrics,
                     model_selection as skms,
                     naive_bayes, neighbors)
import joblib

# we set random_state so the results are reproducable
# otherwise, we get different training and testing sets
# more details in Chapter 5
iris = datasets.load_iris()
(iris_train_ftrs, iris_test_ftrs,
 iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=.15,
                                                        random_state=42)


model = neighbors.KNeighborsClassifier(n_neighbors=3).fit(iris_train_ftrs, iris_train_tgt)


joblib.dump(model, "ml_model_test.joblib")