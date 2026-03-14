
import torch
from momentfm import MOMENTPipeline

class Moment(torch.nn.Module):
    def __init__(self, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MOMENTPipeline 
        self.model_kwargs={
                'task_name': 'classification',
                'n_channels': 6,
                'num_class': 14
            }
    
    def from_pretrained(self, path='AutonLab/MOMENT-1-large'):
        return self.model.from_pretrained(path, model_kwargs=self.model_kwargs)
