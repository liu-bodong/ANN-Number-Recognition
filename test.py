from model import Network
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.Compose([transforms.ToTensor()])
    test_data_set = MNIST("", False, transform=to_tensor, download=True)
    # test_data_loader = DataLoader(data_set, batch_size=32, shuffle=True)

    model = Network().to(device)
    model.load_state_dict(torch.load('mnist.pth', map_location=torch.device('cpu'), weights_only=True))

    correct = 0
    for id, (x, y) in enumerate(test_data_set):
        x = x.to(device)
        y = y.to(device)
        x = x.view(-1, 784)
        pred = (model.forward(x)).argmax(1).item()

        # if id <= 3:
        #     # print("hello?")
        #     plt.figure(id)
        #     plt.imshow(x[0].view(28, 28))
        #     plt.title("prediction: " + str(int(pred)))

        if pred == y:
            correct+=1
    
    sample_num = len(test_data_set)
    acc = correct * 1.0 / sample_num
    print("test accuracy = %d / %d = %.3lf" % (correct, sample_num, acc))


