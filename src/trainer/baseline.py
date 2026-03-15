from aeon.classification.hybrid import RISTClassifier

class BaselineTrainer:
    def __init__(self, network=RISTClassifier(), device="cpu"):
        self.fine_tuned_model = network
        self.is_fitted = False
    
    def fit(self, X, y, *args, **kwargs):
        self.fine_tuned_model.fit(X.numpy().astype('float64'), y.numpy())
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.fine_tuned_model.predict(X.numpy().astype('float64'))