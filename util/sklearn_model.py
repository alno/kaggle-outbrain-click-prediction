

class SklearnModel(object):

    def __init__(self, model):
        self.model = model

    def fit(self, train_X, train_y, train_g=None, eval_X=None, eval_y=None, eval_g=None):
        self.model.fit(train_X, train_y)
        return self

    def predict(self, test_X):
        return self.model.predict_proba(test_X)[:, 1]
