import torch
import torch.nn as nn
import numpy as np
import time
from model import Net
from dataset import DataSet
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Set epoch numbers
EPOCH = 70


# Get transforms of images
def get_transform():
    return transforms.Compose([transforms.Resize((32, 32)), transforms.CenterCrop(24), transforms.ToTensor(),
                               transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))])


# Train one epoch and print out some data
def train_one_epoch(model, data_loader, epoch, device, optimizer, criterion):
    model.train()
    model.to(device)
    start_time = time.time()
    running_loss = 0
    for i, data in enumerate(data_loader, 0):
        img, lbl = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, lbl)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    end_time = time.time()
    print('Epoch {} Training: '.format(epoch), '  Time: {}s'.format(int(end_time-start_time)), '  loss: {}'.format(running_loss))
    return running_loss


# Evaluate one epoch and print out some data
def eval_one_epoch(model, data_loader, device):
    model.eval()
    model.to(device)
    start_time = time.time()
    correct = 0
    wrong = 0
    for i, data in enumerate(data_loader):
        img, lbl = data[0].to(device), data[1].to(device)
        output = model(img)
        for j in range(len(output)):
            predicted_label = torch.argmax(output[j])
            if predicted_label == lbl[j]:
                correct += 1
            else:
                wrong += 1
    end_time = time.time()
    accuracy = float(correct) / (correct + wrong)
    print('Evaluating: ', '  Time: {}s'.format(int(end_time-start_time)), '  Accuracy: ' + '{:.2%}'.format(accuracy))
    return accuracy


# Save the image of Accuracy VS Epoch
def draw(epochs, accuracies):
    plt.figure(num=1, figsize=(8, 8))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch-Accuracy')
    plt.plot(epochs, accuracies, 'b')

    plt.xticks(np.arange(0, EPOCH+2, 5))
    plt.yticks(np.arange(0.7, 1.0, 0.05))
    plt.savefig('output_64batch.png')


if __name__ == '__main__':
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    data_set = DataSet('./GTSRB/Final_Training/Images', transform=get_transform())
    train_set, eval_set = torch.utils.data.random_split(data_set, [int(0.8 * len(data_set)),
                                                                   len(data_set) - int(0.8 * len(data_set))])
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    eval_loader = data.DataLoader(eval_set, batch_size=128, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accuracies = []
    epochs = []

    for epoch in range(EPOCH):
        epochs.append(epoch+1)
        loss = train_one_epoch(model, train_loader, epoch, device, optimizer, criterion)
        accuracy = eval_one_epoch(model, eval_loader, device)
        accuracies.append(accuracy)

    torch.save(model, 'model_64batch.pth')
    draw(epochs, accuracies)
    print('Total Time: {}s'.format(int(time.time()-start)))
