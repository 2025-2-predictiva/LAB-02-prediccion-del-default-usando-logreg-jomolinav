
import os
import pandas as pd

import gzip
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import  SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score

def load_preprocess_data():
    train_path = 'files/input/train_data.csv.zip'
    test_path = 'files/input/test_data.csv.zip'

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    train_dataset.rename(columns={"default payment next month": "default"}, inplace=True)
    test_dataset.rename(columns={"default payment next month": "default"}, inplace=True)

    train_dataset.drop(columns=["ID"], inplace=True)
    test_dataset.drop(columns=["ID"], inplace=True)

    train_dataset = train_dataset.loc[train_dataset["MARRIAGE"] != 0]
    train_dataset = train_dataset.loc[train_dataset["EDUCATION"] != 0]
    test_dataset = test_dataset.loc[test_dataset["MARRIAGE"] != 0]
    test_dataset = test_dataset.loc[test_dataset["EDUCATION"] != 0]

    train_dataset["EDUCATION"] = train_dataset["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    test_dataset["EDUCATION"] = test_dataset["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    train_dataset.dropna(inplace=True)
    test_dataset.dropna(inplace=True)

    return train_dataset, test_dataset

def make_train_test_split(train_dataset, test_dataset):
    x_train = train_dataset.drop(columns=["default"])
    y_train = train_dataset["default"]
    x_test = test_dataset.drop(columns=["default"])
    y_test = test_dataset["default"]

    return x_train, y_train, x_test, y_test

def make_pipeline(x_train):
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = list(set(x_train.columns).difference(categorical_features))

    preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("scaler", MinMaxScaler(), numerical_features),
            ],
            remainder='passthrough'
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', LogisticRegression(random_state=42))
        ],
    )

    return pipeline

def make_grid_search(pipeline, x_train, y_train):
    param_grid = {
    "feature_selection__k": range(1,11),
    "classifier__C": [0.0001, 0.01, 0.1, 1, 10, 100],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ['liblinear'],
    'classifier__max_iter': [100, 200],    
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
    )
    grid_search.fit(x_train, y_train)

    return grid_search

def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)     

def calc_metrics(model, x_train, y_train, x_test, y_test):
    metrics = []

    for x, y, label in [(x_train, y_train, 'train'), (x_test, y_test, 'test')]:
        y_pred = model.predict(x)

        precision = precision_score(y, y_pred, average="binary")
        balanced_acc = balanced_accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred, average="binary")
        f1 = f1_score(y, y_pred, average="binary")

        metrics.append({
            'type': 'metrics',
            'dataset': label,
            'precision': precision,
            'balanced_accuracy': balanced_acc,
            'recall': recall,
            'f1_score': f1
        })
    for x, y, label in [(x_train, y_train, 'train'), (x_test, y_test, 'test')]:
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        metrics.append({
            'type': 'cm_matrix',
            'dataset': label,
            'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
            'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])}
        })

    return metrics

def save_metrics(metrics):
    metrics_path = "files/output"
    os.makedirs(metrics_path, exist_ok=True)
    
    with open("files/output/metrics.json", "w") as file:
        for metric in metrics:
            file.write(json.dumps(metric, ensure_ascii=False))
            file.write('\n')

def main():
    train_dataset, test_dataset = load_preprocess_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_dataset, test_dataset)
    pipeline = make_pipeline(x_train)
    model = make_grid_search(pipeline, x_train, y_train)
    save_estimator(model)
    metrics = calc_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)

if __name__ == "__main__":
    main()