import torch
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder

class LSSTProcessor:
    def __init__(self, target_length=64):
        self.scaler = TimeSeriesScalerMeanVariance()
        self.label_encoder = LabelEncoder()
        self.target_length = target_length
        self._is_fitted = False

    def _prepare_tensor(self, X):
        """
        Interpolation and axes change.

        X: torch.Tensor of chape (batch size, sequence length, channel)
        
        """
        X_t = torch.as_tensor(X, dtype=torch.float32).transpose(1, 2)
        
        X_resampled = torch.nn.functional.interpolate(
            X_t, 
            size=self.target_length, 
            mode='linear', 
            align_corners=False
        )
        return X_resampled

    def fit(self, X, y=None):
        self.scaler.fit(X)
        if y is not None:
            self.label_encoder.fit(y)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        if not self._is_fitted:
            raise RuntimeError("Processor must be fitted before calling transform.")
        
        X_scaled = self.scaler.transform(X)
        
        X_tensor = self._prepare_tensor(X_scaled)
        
        if y is not None:
            y_encoded = torch.tensor(self.label_encoder.transform(y), dtype=torch.long)
            return X_tensor, y_encoded
        
        return X_tensor

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)