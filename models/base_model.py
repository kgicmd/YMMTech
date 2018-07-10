import abc

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score


class BaseModel:

    def __init__(self):
        self.model = None

    @abc.abstractmethod
    def train(self, features, targets):
        print('%s training...' % self.__class__.__name__)

    @abc.abstractmethod
    def predict(self, features):
        print('%s predict...' % self.__class__.__name__)

    @abc.abstractmethod
    def predict_prob(self, features):
        print('%s predict_prob...' % self.__class__.__name__)

    @abc.abstractmethod
    def predict_log_prob(self, features):
        print('%s predict_prob...' % self.__class__.__name__)

    @abc.abstractmethod
    def accuracy_score(self, features, targets):
        print('%s accuracy_score...' % self.__class__.__name__)

    def cross_val_score(self, features, targets):
        print('%s cross_val_score...' % self.__class__.__name__)
        scores = cross_val_score(self.model, X=features, y=targets)
        print('cross_val_score results:\n %s' % scores)
        return scores

    def cross_val_predict(self, features, targets):
        print('%s cross_val_predict...' % self.__class__.__name__)
        scores = cross_val_predict(self.model, X=features, y=targets)
        print('cross_val_predict results:\n%s' % scores)
        return scores

    def metrics_mse(self, features, targets, sample_weight=None):
        print('%s metrics_mse...' % self.__class__.__name__)
        targets_pred = self.predict(features)
        result = mean_squared_error(y_true=targets, y_pred=targets_pred,
                                    sample_weight=sample_weight)
        return result

    def metrics_mae(self, features, targets, sample_weight=None):
        print('%s metrics_mae...' % self.__class__.__name__)
        targets_pred = self.predict(features)
        result = mean_absolute_error(y_true=targets, y_pred=targets_pred,
                                     sample_weight=sample_weight)
        return result

    def feature_ranking(self, feature_names):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for rank, index in enumerate(indices):
            print("Rank %d: %s (%f)" % (rank + 1, feature_names[index], importances[index]))

    def save_model(self, path):
        print('%s save_model...' % self.__class__.__name__)
        joblib.dump(self.model, path)

    def load_model(self, path):
        print('%s load_model...' % self.__class__.__name__)
        self.model = joblib.load(path)
