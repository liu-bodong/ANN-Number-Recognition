from model import Network
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.Compose([transforms.ToTensor()])
    test_data_set = MNIST("", False, transform=to_tensor, download=True)

    model = Network().to(device)
    model.load_state_dict(torch.load('mnist.pth', map_location=torch.device('cpu'), weights_only=True))

    correct = 0
    for id, (x, y) in enumerate(test_data_set):
        x = x.to(device)
        x = x.view(-1, 784)
        pred = model(x)
        _, pred = torch.max(pred.data, 1)
        pred = pred.cpu().numpy()
        correct += (pred == y).sum()
    
    sample_num = len(test_data_set)
    acc = correct * 1.0 / sample_num
    return acc

def evaluate(model, test_data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_data_loader:
            x = x.view(-1, 784)
            pred_y = model(x)
            _, pred = torch.max(pred_y.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

if __name__ == '__main__':
    acc = test()
    print(f'accuracy: {acc:.4f}')
