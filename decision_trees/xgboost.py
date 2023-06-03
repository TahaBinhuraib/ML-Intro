import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class XGBoostClassifier:
    def __init__(self, num_boost_round=10, early_stopping_rounds=5):
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {"objective": "binary:logistic", "eval_metric": "error", "seed": 42}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dval, "validation")],
        )

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        y_pred = self.model.predict(dtest)
        y_pred_binary = [round(value) for value in y_pred]
        return y_pred_binary

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy
