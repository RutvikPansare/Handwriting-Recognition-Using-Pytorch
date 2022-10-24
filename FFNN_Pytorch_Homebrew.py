import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle




INPUT_LAYER = 50

class handWritingDataset(Dataset):
    def __init__(self):
        file = open('./Data/Subsets/DAT_Representation/dataset_final.dat', 'rb+')
        data = pickle.load(file)
        file.close()
        rawimages = data[0]
        labels = data[1]
        images = []
        for i in range(len(rawimages)):
            temp = np.array(rawimages[i]).flatten().tolist()
            images.append(torch.tensor(temp, dtype=torch.float64))
        self.X = torch.vstack(images)
        self.Y = torch.tensor(labels, dtype=torch.float64)
        self.n_samples = self.X.size(0)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.n_samples

def retrieveHandWritingData(train_percent=0.7, validation_percent=0.10):
    dataset = handWritingDataset()
    ds = len(dataset)
    train_len = int(ds * (train_percent))
    validation_len = int(ds * validation_percent)
    test_len = ds - train_len - validation_len

    train_set, test_set, valid_set = torch.utils.data.random_split(dataset, [train_len, test_len, validation_len])

    return train_set, test_set, valid_set


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 50)
        # self.fc2 = nn.Linear(101, 101)
        self.fc3 = nn.Linear(50, 30)
        self.fc4 = nn.Linear(30, 30)
        # self.fc5 = nn.Linear(30, 30)
        # self.fc6 = nn.Linear(30, 30)
        self.fc7 = nn.Linear(30, 25)



    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.long)
        # print(x.type())
        x = torch.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        return x


def FFNN(dataset, net, train=True, EPOCHS=20, batch_size=500, learning_rate=0.0003, momentum=0.9, gamma=0.9, device=None):

    # loss_class = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()
    if device is not None:
        loss_class.to(device)
        net.to(device)
    if train:
        # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        EPOCHS = 1

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(EPOCHS):
        outputs = torch.empty((0, 25), requires_grad=False)
        targets = torch.empty((0, 25), requires_grad=False)
        outputs = outputs.to(device)
        targets = targets.to(device)
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            output = net(data)
            loss = loss_class(output, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            outputs = torch.cat((outputs, output), 0)
            targets = torch.cat((targets, labels), 0)
        outputs = np.array(outputs.to('cpu').tolist())
        targets = np.array(targets.to('cpu').tolist())
        if train:
            scheduler.step()
        pseudo_loss = np.round(np.sum(np.argmax(outputs, 1) == np.argmax(targets, 1)) / outputs.shape[0], 4)
        printloss = torch.trunc(torch.round(loss*1000))/1000

        if train:
            print(f'Loss of FFNN: {printloss}  -  epoch {epoch + 1}  -   Accuracy {pseudo_loss}')
        else:
            print(f'Loss of FFNN: {printloss}  -  Accuracy {pseudo_loss}')

def FFNN_RSIN(dataset, net):
    output = net(dataset)
    return output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    net = Net()
    net = net.double()
    print(net)

    train_set, test_set, _ = retrieveHandWritingData(validation_percent=0)
    batch_size = 50
    EPOCHS = 25
    learning_Rate = 0.003
    momentum = 0.9
    gamma = 0.5

    FFNN(train_set, net,
         train=True,
         EPOCHS=EPOCHS,
         batch_size=batch_size,
         learning_rate=learning_Rate,
         momentum=momentum,
         gamma=gamma,
         device=device
         )
    FFNN(test_set, net,
         train=False,
         EPOCHS=EPOCHS,
         batch_size=batch_size,
         learning_rate=learning_Rate,
         momentum=momentum,
         gamma=gamma,
         device=device
         )

    # saveData(net, 'FFNN_NetClass.dat')

    # file = open('./Data/Raw_RS_Data/datafile1.dat', 'rb')
    # data = pickle.load(file)
    # ms = data[0]
    # rs = RandomScene.randomScene(silent=True)
    # print(rs.mean, rs.std)
    # scene_utilities.sceneSurf(rs)

    file = open('FFNN_Model.dat', 'wb+')
    pickle.dump(net, file)
    file.close()








if __name__ == '__main__':
    main()