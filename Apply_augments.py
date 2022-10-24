import cv2
import numpy as np
import augments
import augments2
import pickle
import copy

file = open('./Data/Subsets/DAT_Representation/dataset_centered_downsampled.dat', 'rb')
data = pickle.load(file)
f2 = open('./Data/Subsets/DAT_Representation/scan_processed.dat', 'rb+')
scanned_dataset = pickle.load(f2)

images = np.concatenate((data[0], scanned_dataset[0]), axis=0)
labels = data[1] + scanned_dataset[1]
print(np.shape(images[0]), type(labels))
# cv2.imshow('asdf', cv2.resize(images[0], (560,560)))
# cv2.waitKey(1000)

new_images = copy.deepcopy(images[:, :, :, 0])
new_labels = copy.deepcopy(labels)
for i in range(25):
    print('Iteration:', i+1)
    for j in range(images.shape[0]):
        if i == 0:
            #rotate 15 degrees
            im = augments.imrotBW(images[j], 15)
        elif i == 1:
            #rotate -15 degrees
            im = augments.imrotBW(images[j], -15)
        elif i == 2:
            # rotate 30 degrees
            im = augments.imrotBW(images[j], 30)
        elif i == 3:
            # rotate -30 degrees
            im = augments.imrotBW(images[j], -30)
        elif i == 4:
            # rotate 45 degrees
            im = augments.imrotBW(images[j], 45)
        elif i == 5:
            # rotate -45 degrees
            im = augments.imrotBW(images[j], -45)
        elif i == 6:
            # skew 10 degrees, option 1
            im = augments.skew_mat(images[j], 1, 0)
        elif i == 7:
            # rotate -15 degrees
            im = augments.skew_mat(images[j], -1, 0)
        elif i == 8:
            # skew 10 degrees, option 1
            im = augments.skew_mat(images[j], 2, 0)
        elif i == 9:
            # rotate -15 degrees
            im = augments.skew_mat(images[j], -2, 0)
        elif i == 10:
            # rotate 30 degrees
            im = augments.skew_mat(images[j], 3, 0)
        elif i == 11:
            # rotate -30 degrees
            im = augments.skew_mat(images[j], -3, 0)
        elif i == 12:
            # rotate 45 degrees
            im = augments.skew_mat(images[j], 1, 2)
        elif i == 13:
            # rotate -45 degrees
            im = augments.skew_mat(images[j], -1, 2)
        elif i == 14:
            # rotate 45 degrees
            im = augments.skew_mat(images[j], 2, 2)
        elif i == 15:
            # rotate -45 degrees
            im = augments.skew_mat(images[j], -2, 2)
        elif i == 16:
            # asdf
            im = augments.skew_mat(images[j], 3, 2)
        elif i == 17:
            # skew 10 degrees, option 1
            im = augments.skew_mat(images[j], -3, 2)

        im = np.reshape(im, (1, 28, 28))
        new_images = np.concatenate((new_images, im), axis=0)
        new_labels.append(copy.deepcopy(labels[j]))

new_data = [new_images, new_labels]

file.close()

newfile = open('./Data/Subsets/DAT_Representation/dataset_augmented.dat', 'wb+')
pickle.dump(new_data, newfile)

newfile.close()