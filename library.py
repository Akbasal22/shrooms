import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data import get_x_y


splits, test_set = get_x_y('./processed.csv')


x_train, y_train, _, _ = splits[3]
x_test, y_test = test_set

#xgboost needs numeric values
y_train=np.where(y_train=='e',0 ,1)
y_test=np.where(y_test=='e',0 ,1)

model=xgb.XGBClassifier(use_label_encode=False,eval_metric='logloss')
model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(f"accuracy: {accuracy * 100:.2f}%")