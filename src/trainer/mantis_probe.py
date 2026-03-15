from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mantis.trainer import MantisTrainer

class MantisProbeTrainer:
    def __init__(self, network, device="cpu"):
        self.model_trainer = MantisTrainer(network=network, device=device)
        self.fine_tuned_model = self.model_trainer.network
        self.is_fitted = False
        self.cls = LogisticRegression(C=0.1, max_iter=1000)
        self.scaler = StandardScaler()

    
    def fit(self, X, y, *args, **kwargs):
        Z = self.model_trainer.transform(X)
        Z = self.scaler.fit_transform(Z)
        self.cls.fit(Z, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        Z = self.model_trainer.transform(X)
        Z = self.scaler.transform(Z)
        return self.cls.predict(Z)