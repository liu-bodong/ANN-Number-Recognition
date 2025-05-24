import torch
from model import Network
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

if __name__ == '__main__':
    print("training started")
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", True, transform=to_tensor, download=True)
    train_data_loader = DataLoader(data_set, batch_size=16, shuffle=True)

    model = Network()
    optimizer = optim.Adam(model.parameters())  
    criterion = nn.CrossEntropyLoss() 

    print("data loaded")

    for epoch in range(10):
        losses = torch.zeros(0)
        for batch_id, (x, y) in enumerate(train_data_loader):
            optimizer.zero_grad()
            x = x.view(-1, 784)
            pred_y = model.foward(x)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            losses = torch.cat((losses, loss.detach().unsqueeze(0)), 0)
        mean_loss = np.mean(losses.detach().numpy())
        # print(f'epoch={epoch}, mean loss={mean_loss}')
        # evaluate
        # print(f'epoch={epoch}, accuracy={}')

    torch.save(model.state_dict(), 'mnist.pth')






