import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle




INPUT_LAYER = 50

class handWritingDataset(Dataset):
    def __init__(self, force_rgb=False, dtype=torch.float64):
        file = open('./Data/Subsets/DAT_Representation/dataset_final.dat', 'rb+')
        data = pickle.load(file)
        file.close()
        rawimages = data[0]
        labels = data[1]
        images = []
        for i in range(len(rawimages)):
            temp = np.reshape(np.array(rawimages[i]), (1, 1, 28, 28))
            if force_rgb:
                temp = np.concatenate((temp, temp, temp), 1).tolist()
            images.append(torch.tensor(temp, dtype=dtype))
        self.X = torch.vstack(images)
        self.Y = torch.tensor(labels, dtype=dtype)
        self.n_samples = self.X.size(0)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.n_samples

def retrieveHandWritingData(train_percent=0.7, validation_percent=0.10, force_rgb=False, dtype=torch.float64):
    dataset = handWritingDataset(force_rgb=force_rgb, dtype=dtype)
    ds = len(dataset)
    train_len = int(ds * (train_percent))
    validation_len = int(ds * validation_percent)
    test_len = ds - train_len - validation_len

    train_set, test_set, valid_set = torch.utils.data.random_split(dataset, [train_len, test_len, validation_len])

    return train_set, test_set, valid_set


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0
        )

        # self.conv2 = nn.Conv2d(
        #     in_channels=32,
        #     out_channels=32,
        #     kernel_size=5,
        #     stride=1,
        #     padding=0
        # )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.fc1 = nn.Linear(
            # in_features=18 * 18 * 128,
            in_features=3200, #32 * 44 * 44,
            out_features=256
        )

        self.Dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(
            in_features=256,
            out_features=25
        )



    def forward(self, x):
        # Relu activation function
        x = torch.relu(self.conv1(x))
        # Relu activation function
        # x = torch.relu(self.conv2(x))
        # Max pooling layer, kernel size of 2, stride of 2
        x = torch.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
        # Relu activation function, on maxpooling layer, assuming that relu must be applied to internal ReLu layer
        x = torch.relu(x)
        # print(x.shape)
        x = x.view(-1, 3200)
        # Relu activation function
        x = torch.relu(self.fc1(x))
        # Dropout on fully connected layer
        x = self.Dropout(x)
        # Output layer
        x = torch.sigmoid(self.fc2(x))
        return x


def CNN(dataset, net, train=True, EPOCHS=20, batch_size=500, learning_rate=0.0003, momentum=0.9, gamma=0.9, device=None, optimized_params=None):

    # loss_class = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()
    if device is not None:
        loss_class.to(device)
        net.to(device)
    if train:
        # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
        if optimized_params is None:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        else:
            optimizer = optim.SGD(optimized_params, lr=learning_rate, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
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
    if device is not None:
        loss_class.to('cpu')
        net.to('cpu')

def CNN_RSIN(dataset, net):
    output = net(dataset)
    return output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    net = Net()
    net = net.double()
    print(net)

    train_set, test_set, _ = retrieveHandWritingData(validation_percent=0)
    batch_size = 200
    EPOCHS = 25
    learning_Rate = 0.003
    momentum = 0.9
    gamma = 0.5

    CNN(train_set, net,
        train=True,
        EPOCHS=EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_Rate,
        momentum=momentum,
        gamma=gamma,
        device=device
        )
    torch.cuda.empty_cache()
    # FFNN(test_set, net,
    #      train=False,
    #      EPOCHS=EPOCHS,
    #      batch_size=batch_size,
    #      learning_rate=learning_Rate,
    #      momentum=momentum,
    #      gamma=gamma,
    #      device=device
    #      )

    # saveData(net, 'FFNN_NetClass.dat')
    #
    # file = open('./Data/Raw_RS_Data/datafile1.dat', 'rb')
    # data = pickle.load(file)
    # ms = data[0]
    # rs = RandomScene.randomScene(silent=True)
    # print(rs.mean, rs.std)
    # scene_utilities.sceneSurf(rs)

    file = open('CNN_Model.dat', 'wb+')
    pickle.dump(net, file)
    file.close()








if __name__ == '__main__':
    main()