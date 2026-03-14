import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MomentTrainer:
    def __init__(self, network, device):
        self.fine_tuned_model = network
        self.fine_tuned_model.init()
        self.fine_tuned_model = self.fine_tuned_model.to(device)
        self.device = device
    def fit(self, X, y, num_epochs, fine_tuning_type, init_optimizer):
        dataset = TensorDataset(X, y)
        train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True) 
        criterion = nn.CrossEntropyLoss()
        optimizer = init_optimizer(self.fine_tuned_model.parameters())

        pbar = tqdm(range(num_epochs))

        for epoch in pbar:
            total_loss = 0
            for data, labels in train_dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                output = self.fine_tuned_model(x_enc=data)
                loss = criterion(output.logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            pbar.set_postfix({f"Epoch {epoch+1} Loss": f"{total_loss/len(train_dataloader):.4f}"})
        
        return self
        
    @torch.no_grad
    def predict(self, X):
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False) 
        y_pred = []
        self.fine_tuned_model.eval()
        for [data] in dataloader:
            data = data.to(self.device)
            output = self.fine_tuned_model(x_enc=data)
            output = torch.argmax(output.logits, dim=1).cpu()
            y_pred.append(output)
        return torch.cat(y_pred, 0)


    