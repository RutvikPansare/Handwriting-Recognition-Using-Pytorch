import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
# from pytorch_FFNN_example import Net, retrieveHandWritingData
from CNN_PyTorch_Homebrew import Net
import CNN_PyTorch_Homebrew

def loadModel(model_no):
    if model_no == 1:
        filepath = './FFNN_Model.dat'
    elif model_no == 2:
        filepath = './CNN_Model.dat'
    elif model_no == 3:
        filepath = './found_pytorch_model.dat'
    elif model_no == 4:
        filepath = './resnet_model.dat'
    else:
        filepath = './FFNN_Model.dat'
    file = open(filepath, 'rb+')
    model = pickle.load(file)
    return model

def prettySquare(mat):
    for i in range(mat.shape[1]):
        temp = '['
        for j in range(mat.shape[0]):
            sn = str(mat[i,j])
            ss = len(sn)
            temp = temp + (4-ss)*' ' + sn
        temp = temp + ']'
        print(temp)

def main():
    train_set, temp, temp2 = CNN_PyTorch_Homebrew.retrieveHandWritingData(train_percent=1, validation_percent=0,
                                                                          force_rgb=False, dtype=torch.float64)
    # train_set_rgb, _, _ = pytorch_CNN_example.retrieveHandWritingData(train_percent=1, validation_percent=0,
    #                                                                              force_rgb=True, dtype=torch.float)

    y = torch.empty((0, 25), requires_grad=False)
    x = torch.empty((0, 1, 28, 28), requires_grad=False)
    dataloader = DataLoader(dataset=train_set, batch_size=5000, shuffle=True, num_workers=4)
    for i, (data, labels) in enumerate(dataloader):
        y = torch.cat((y, labels.view((-1, 25))), 0)
        x = torch.cat((x, data), 0)
    print(x.size())
    x.to('cpu')
    y.to('cpu')
    # for i in [1]:
    model = loadModel(1).to('cpu')
    outputs = model(x)
    ynp = np.array(y.tolist())
    onp = np.array(outputs.tolist())
    conf = np.zeros((25, 25))
    for j in range(ynp.shape[0]):
        # for k in range()
        yind = np.argmax(ynp[j])
        oind = np.argmax(onp[j])
        conf[yind, oind] = conf[oind, yind] + 1
    print('Confusion Matrix for Model')
    prettySquare(conf.astype(np.uint8))

if __name__ == '__main__':
    main()