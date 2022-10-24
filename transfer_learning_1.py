import pickle

import torch
import torch.nn as nn
# Line Below needed for pickle to work right
from Pytorch_MNIST_Online import Net
from torchvision import datasets, models, transforms
import CNN_PyTorch_Homebrew


def main():
    train_resnet = False
    if train_resnet:
        model = models.resnet18(pretrained=True)
        train_set, test_set, _ = CNN_PyTorch_Homebrew.retrieveHandWritingData(validation_percent=0, force_rgb=True,
                                                                              dtype=torch.float)
        model.fc = nn.Linear(model.fc.in_features, 25)
        optim_param = model.fc.parameters()
    else:
        file = open('torch_mnist_model.dat', 'rb+')
        model = pickle.load(file)
        train_set, test_set, _ = CNN_PyTorch_Homebrew.retrieveHandWritingData(validation_percent=0, force_rgb=False,
                                                                              dtype=torch.float)
        model.fc2 = nn.Linear(model.fc2.in_features, 25)
        optim_param = model.fc2.parameters()

    batch_size = 200
    EPOCHS = 6
    learning_Rate = 0.003
    momentum = 0.9
    gamma = 0.5

    # print(len(test_set))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNN_PyTorch_Homebrew.CNN(train_set, model,
                             train=True,
                             EPOCHS=EPOCHS,
                             batch_size=batch_size,
                             learning_rate=learning_Rate,
                             momentum=momentum,
                             gamma=gamma,
                             device=device,
                             optimized_params=optim_param
                             )
    CNN_PyTorch_Homebrew.CNN(test_set, model,
                             train=False,
                             EPOCHS=EPOCHS,
                             batch_size=batch_size,
                             learning_rate=learning_Rate,
                             momentum=momentum,
                             gamma=gamma,
                             device='cpu'
                             )

    if train_resnet:
        file = open('resnet_model.dat', 'wb+')
        pickle.dump(model, file)
    else:
        file = open('found_pytorch_model.dat', 'wb+')
        pickle.dump(model, file)

if __name__ == '__main__':
    main()

