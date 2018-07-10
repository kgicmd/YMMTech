import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """RandomForest classifier."""

    def __init__(self,
                 n_estimators=100,  # 树的数量
                 max_features='sqrt',  # 每次随机选择特征子集的大小
                 criterion='gini',  # 使用基尼指数作为特征选择标准
                 class_weight=None,  # 类别权重
                 max_depth=None,  # 树的深度, None表示知道所有叶子节点的样本完全纯净或者数量小于min_samples_split
                 min_samples_split=2,  # 节点中最小样本数量
                 bootstrap=True,  # 采用自助采样
                 oob_score=True,  # 采用包外估计
                 n_jobs=1,
                 verbose=1):
        super(RandomForestModel).__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,  # 树的数量
            max_features=max_features,  # 每次随机选择特征子集的大小
            criterion=criterion,  # 使用基尼指数作为特征选择标准
            class_weight=class_weight,  # 类别权重
            max_depth=max_depth,  # 树的深度, None表示知道所有叶子节点的样本完全纯净或者数量小于min_samples_split
            min_samples_split=min_samples_split,  # 节点中最小样本数量
            bootstrap=bootstrap,  # 采用自助采样
            oob_score=oob_score,  # 采用包外估计
            n_jobs=n_jobs,
            verbose=verbose)

    def predict(self, features):
        super().predict(features)
        labels = self.model.predict(features)
        return labels

    def predict_prob(self, features):
        super().predict_prob(features)
        probs = self.model.predict_proba(features)
        return probs

    def predict_log_prob(self, features):
        super().predict_log_prob(features)
        probs = self.model.predict_log_proba(features)
        return probs

    def predict_individuals(self, features):
        """
        使用随机森林中的每个分类器进行单独的预测，产生预测结果.
        :param features: 待预测的特征
        :return: 每个单独的分类器的预测结果
        """
        print('%s predict_individuals.' % self.__class__.__name__)
        y_hats = []
        for e in self.model.estimators_:
            y_hats.append(e.predict(features))
        return np.array(y_hats)

    def train(self, features, targets):
        super().train(features, targets)
        start = time.time()
        self.model.fit(X=features, y=targets)
        print('Finished, time %s' % (time.time() - start))

    def accuracy_score(self, features, targets):
        super().accuracy_score(features, targets)
        score = self.model.score(features, targets, self.model.class_weight)
        return score

    def abs_errors(self, features, targets):
        targets_pred = self.predict(features)
        result = abs(targets_pred - targets)
        return result

    def rmse_score(self, y_pred, y_true):
        """
        计算RMSE评分，为了体现预测结果0、1、2不同的重要性，增加对1,2预测错误的惩罚度，
        在评分计算时对不同行为分别乘以1,2,2.5的权重因子。
        np.average((y_true - y_pred) ** 2, axis=0, weights=weights)
        :param y_pred: 预测标签
        :param y_true: 真实标签
        :return: 评分
        """
        weight_dict = {0: 1, 1: 2, 2: 2.5}  # 不同类别的误判惩罚权重
        weights = [weight_dict[l] for l in y_true]
        mse = np.average((y_true - y_pred) ** 2, axis=0, weights=weights)
        score = 1 / (1 + np.sqrt(mse))
        return score


class Dataset:

    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


def split_dataset(data_x, data_y, test_size=0.2, random_state=0):
    """
    Split dataset into train set and test set.
    :param random_state:
    :param data_x:
    :param data_y:
    :param test_size:
    :rtype Dataset
    """
    sss = StratifiedShuffleSplit(2, test_size=test_size, random_state=random_state)
    (train_indices, test_indices), _ = sss.split(data_x, data_y)
    return Dataset(data_x.values[train_indices], data_y.values[train_indices],
                   data_x.values[test_indices], data_y.values[test_indices])


if __name__ == '__main__':
    _train_features_df = pd.read_csv('../data/preprocess/train_features.csv')
    _train_features_df = _train_features_df.drop(columns=['user_id', 'id'])

    _train_x = _train_features_df.drop(columns='label')
    _train_y = _train_features_df.loc[:, 'label']
    _dataset = split_dataset(_train_x, _train_y)

    # rf = RandomForestModel(n_estimators=200,
    #                        min_samples_split=5,
    #                        class_weight='balanced')
    # rf.train(_train_x, _train_y)
    # rf.save_model('../ckpts/random_forest_model.ckpt')

    rf = RandomForestModel()
    rf.load_model('../ckpts/random_forest_model.ckpt')

    rf.feature_ranking(list(_train_x.columns))

    # pred_prob = rf.predict_prob(_dataset.test_x)
    # pred_labels = [2 if p[2] > 0.7 else 1 if p[1] - p[0] > 0.1 else 0 for p in pred_prob]

    # pred_labels = rf.predict(_dataset.test_x)

    # rmse_score = rf.rmse_score(pred_labels, _dataset.test_y)
    # print('RMSE score: %s' % rmse_score)

    # Prediction.
    _test_features_df = pd.read_csv('../data/preprocess/predict_features.csv')
    _test_features_df = _test_features_df.drop(columns=['user_id', 'id'])

    _test_x = _test_features_df.drop(columns='label')

    _pred_prob = rf.predict_prob(_test_x)
    _pred_labels = [2 if p[2] > 0.5 else 1 if p[1] > p[0] else 0 for p in _pred_prob]

    _predict = pd.read_csv('../data/predict_data/predict.csv', header=None)
    _predict_res = pd.DataFrame({'driver_id': _predict.iloc[:, 0],
                                 'cargo_id': _predict.iloc[:, 1],
                                 'score': _pred_labels})
    _predict_res.to_csv('../data/predict_data/predict_result.csv', header=None, index=False)
