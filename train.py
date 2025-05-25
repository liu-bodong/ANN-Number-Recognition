import torch
from model import Network
from test import evaluate
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

def train(debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    to_tensor = transforms.Compose([transforms.ToTensor()])
    train_data_set = MNIST("", True, transform=to_tensor, download=True)
    train_data_loader = DataLoader(train_data_set, batch_size=64, shuffle=True)

    model = Network().to(device)
    optimizer = optim.Adam(model.parameters())  
    criterion = nn.CrossEntropyLoss() 
    for epoch in range(20):
        for batch_id, (x, y) in enumerate(train_data_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            x = x.view(-1, 784)
            pred_y = model.forward(x)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            if debug and batch_id == 0 and epoch % 10 == 0:
                print(f'epoch: {epoch}, batch id: {batch_id}, loss: {loss.item()}')
                # acc = evaluate(model, train_data_loader)
                # print(f'epoch: {epoch}, batch id: {batch_id}, acc: {acc:.4f}')
            
    return model

if __name__ == '__main__':
    print("training...")
    trained_model = train(debug=False)

    torch.save(trained_model.state_dict(), 'mnist.pth')






