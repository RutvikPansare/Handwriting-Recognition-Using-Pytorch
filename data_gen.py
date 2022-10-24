import pickle
import numpy as np
from pdf2image import convert_from_path
import augments
from PIL import Image, ImageOps
import queue
from matplotlib import pyplot as plt
import copy
import cv2
import scipy.interpolate
from matplotlib import pyplot as plt


def convPDF2JPG(people, types):
    pdfs = []
    for i in people:
        for j in types:
            pdfs.append('./Data/Raw/PDF_Representation/Fat/' + j + i + '.pdf')
    print(pdfs)
    for pdf in pdfs:
        # Store Pdf with convert_from_path function
        images = convert_from_path(pdf)
        # Save pages as images in the pdf
        s = np.size(images[0])
        output = cv2.resize(augments.im2nparr(images[0]), (int(s[0] * 1), int(s[1] * 1)))
        fin = augments.nparr2im(output)
        # cv2.imwrite('./Data/Raw/' + pdf[11:-4] + '.jpg', 'JPEG', fin)
        fin.save('./Data/Raw/JPG_Representation/Raw/' + pdf[34:-4] + '.png', 'PNG')


def grid_remove(image, ismat=False, plot_image=False):
    if ismat:
        image = augments.im2nparr(image)
    q = queue.Queue()
    s = np.shape(image)
    b = False
    for i in range(s[0]):
        for j in range(s[1]):
            # print(image[i, j, 0])
            if image[i, j, 0] == 0:
                b = True
                break
        if b:
            break
    ij = (copy.copy(i), copy.copy(j))
    q.put(ij)
    visited = [ij]
    ir = [-1, 0, 1]
    jr = [-1, 0, 1]
    while not q.empty():
        item = q.get()
        # print(s, s[0]*s[1], len(visited))
        istart = item[0]
        jstart = item[1]
        for i in ir:
            inew = i + istart
            if inew < s[0]:
                for j in jr:
                    jnew = j + jstart
                    if jnew < s[1]:
                        # print(image[inew, jnew, 0], image[inew, jnew, 0] == 255)
                        if (image[inew, jnew, 0] == 0) and (not ((inew, jnew) in visited)):
                            nij = (copy.copy(inew), copy.copy(jnew))
                            visited.append(nij)
                            # print(visited)
                            q.put(nij)

    image2 = copy.deepcopy(image)
    for i in visited:
        image2[i[0], i[1]] = 255
    if plot_image:
        plt.subplot(1, 2, 1)
        plt.imshow(image, 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(image2, 'gray')
        plt.show()
    return image2


def main():
    people = ['aq', 'bc', 'bk', 'cq', 'mt', 'nq', 'ra']
    types = ['Devanagari_', 'Western_Arabic_']
    rows = 14
    columns = 10

    # option = 1
    for option in [1, 2, 3, 4, 5, 6]:
        # Convert PDFs to images
        if option == 1:
            convPDF2JPG(people, types)
        # Threshold the images
        elif option == 2:
            for i in people:
                th = 100
                for j in types:
                    augments.threshold_image('./Data/Raw/JPG_Representation/Raw/' + j + i + '.png',
                                             thresh=th, open_image=True,
                                             filepath_dat='./Data/Raw/DAT_Representation/Thresholded/' + j + i + '_g.dat',
                                             filepath_image='./Data/Raw/JPG_Representation/Thresholded/' + j + i + '_g.png')
        # Remove the Grid
        elif option == 3:
            for i in people:
                for j in types:
                    filepath_from = './Data/Raw/DAT_Representation/Thresholded/' + j + i + '_g.dat'
                    filepath_to_JPG = './Data/Raw/JPG_Representation/Grid_Removed/' + j + i + '_gr.png'
                    filepath_to_DAT = './Data/Raw/DAT_Representation/Grid_Removed/' + j + i + '_gr.dat'
                    image = pickle.load(open(filepath_from, 'rb+'))
                    image2 = grid_remove(image, plot_image=False)
                    thresh_im = augments.nparr2im(image2)
                    thresh_im.save(filepath_to_JPG, 'JPEG')
                    pickle.dump(image2, open(filepath_to_DAT, 'wb+'))
                    print('Finished:', j + i)
        # Separate the subsets from the images
        elif option == 4:
            x_off = 0
            y_off = 50*4
            dx = 37*4
            dy = 37*4
            subset = []
            labels = []
            for i in people:
                for j in types:
                    filepath_from_DAT = './Data/Raw/DAT_Representation/Grid_Removed/' + j + i + '_gr.dat'
                    filepath_to_JPG_ = './Data/Subsets/JPG_Representation/' + j + i
                    data = pickle.load(open(filepath_from_DAT, 'rb'))
                    s = np.shape(data)
                    for k in range(rows):
                        for l in range(columns):
                            x_cur_m = x_off + l * dx
                            x_cur_p = x_off + (l + 1) * dx# + int(l*2)
                            if x_cur_p > s[1]:
                                x_cur_p = s[1]
                            y_cur_m = y_off + k * dy
                            y_cur_p = y_off + (k + 1) * dy + 1
                            if y_cur_p > s[0]:
                                y_cur_p = s[0]
                            item = data[y_cur_m:y_cur_p, x_cur_m:x_cur_p]
                            subset.append(item)
                            labels.append([j[0], l])
                            item_img = augments.nparr2im(item)
                            filepath_to_JPG = filepath_to_JPG_ + '_' + str(k+1) + '_' + str(l) + '.png'
                            item_img.save(filepath_to_JPG)
            data_out = [subset, labels]
            pickle.dump(data_out, open('./Data/Subsets/DAT_Representation/dataset.dat', 'wb+'))
        # Center the images
        elif option == 5:
            file_path = './Data/Subsets/DAT_Representation/dataset.dat'
            file = open(file_path, 'rb')
            data = pickle.load(file)
            temp = data[0]
            labels = data[1]
            data2 = []
            for i in range(len(temp)):
                d = temp[i]
                s = np.shape(d)
                s2 = (s[0], s[1])

                # print(s, s2)
                rs = d[:, :, 0]
                bs = d[:, :, 1]
                gs = d[:, :, 2]
                r = rs.flatten()
                b = bs.flatten()
                g = gs.flatten()
                white_ind = np.all(d == 255, 2)
                # white_n = np.sum(np.sum(white_ind))
                nw_ind = np.logical_not(white_ind)
                nw_n = np.sum(np.sum(nw_ind))
                x = np.arange(0, s[1], 1)
                y = np.arange(0, s[0], 1)
                xx, yy = np.meshgrid(x, y)

                xm = np.sum(np.sum(xx[nw_ind])) / nw_n
                ym = np.sum(np.sum(yy[nw_ind])) / nw_n
                xd = xm - s[1]/2
                yd = ym - s[0]/2

                xl = xx.flatten()
                yl = yy.flatten()

                xln = xl + xd
                yln = yl + yd

                rn = scipy.interpolate.griddata((xl, yl), r, (xln, yln), fill_value=255).astype(np.uint8)

                rng = np.reshape(rn, s2)

                d2 = np.zeros(s)
                d2[:, :, 0] = rng
                d2[:, :, 1] = rng
                d2[:, :, 2] = rng

                data2.append(d2)

            data_out = [data2, labels]
            pickle.dump(data_out, open('./Data/Subsets/DAT_Representation/dataset_centered.dat', 'wb+'))
        # Downsample the image to MNIST image size (28x28)
        elif option == 6:
            file_path = './Data/Subsets/DAT_Representation/dataset_centered.dat'
            file = open(file_path, 'rb')
            data = pickle.load(file)
            temp = data[0]
            labels = data[1]
            data2 = []
            s = np.shape(temp)
            print(s)
            for i in range(len(temp)):
                d = temp[i]
                d2 = cv2.resize(d, (28,28))
                data2.append(d2)
            data2 = np.array(data2)

            data_out = [data2, labels]
            pickle.dump(data_out, open('./Data/Subsets/DAT_Representation/dataset_centered_downsampled.dat', 'wb+'))
        print('Option', i)

if __name__ == '__main__':
    main()
