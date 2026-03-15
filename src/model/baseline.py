from aeon.classification.hybrid import RISTClassifier

class BaselineModel:
    def __init__(self, device="cpu", *args, **kwargs):
        self.model = RISTClassifier()

    def from_pretrained(self,  *args, **kwargs):
        return self.model