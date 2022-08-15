import torch
from torch import optim
from transformers import VisionEncoderDecoderModel

from src.utils import save_model


device = 'gpu: 0' if torch.cuda.is_available() else 'cpu'
model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def update_learning_rate(lr):
    """Initialize hyperparameters with values given as arguments"""
    for g in optimizer.param_groups:
        g['lr'] = lr

    return optimizer

class Trainer:
    def __init__(self, train_loader, args):
        self.train_loader = train_loader
        self.lr = args.lr
        self.epochs = args.epochs
        self.run_name = args.run_name

    def trainOne(self):
        """To train on one epoch"""
        train_loss = 0
        for _ in range(1):
            for idx, data in enumerate(self.train_dataloader):
                data = data.to(device)
                out = model(data)

                loss = out.backward()

                optimizer.zero_grad()
                optimizer.step()
            
            train_loss += loss.items()
        save_model(model, self.run_name, epoch=0, cloud_storage=False, metrics_required=True)

    def train(self):
        """To train on all epochs"""
        optimizer = update_learning_rate(self.lr)
        for epoch in range(self.epochs):
            pass

        
        """
        if accuracy improved, cloud_storage=True
        """
        save_model(model, self.run_name, epoch, cloud_storage=False, metrics_required=True)
