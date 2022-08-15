import torch
from torch import optim
from transformers import VisionEncoderDecoderModel

device = 'gpu: 0' if torch.cuda.is_available() else 'cpu'
model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def load_hyperparameters(args):
    """Initialize hyperparameters with values given as arguments"""
    pass

def trainOne(train_dataloader):
    train_loss = 0
    """To train on one epoch"""
    for _ in range(1):
        for idx, data in enumerate(train_dataloader):
            data = data.to(device)
            out = model(data)

            loss = out.backward()

            optimizer.zero_grad()
            optimizer.step()
        
        train_loss += loss.items()



def train(train_dataloader):
    """To train on all epochs"""
    pass