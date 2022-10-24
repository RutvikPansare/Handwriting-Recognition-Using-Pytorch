
"""
 preprocessor.py

Author:     Matthew Tighe <mjt6124@rit.edu> \n
Purpose:    Preprocessing functionalities:
                seg2mat(filename)   - Converts a segmented jpg image to a matrix, in the form listed. \n
                split(filename)     - Splits a jpeg image into individually handwritten digits. \n
                segment(filename)   - Segments an image into true-black and true-white pixels. \n
                \n
            Preprocessing Pipe____________________\n
                im2Binary -> floodGrid -> segment \n
"""
import copy
import numpy as np
import pickle
import cv2
import scipy.interpolate


DAT_FILENAME = 'Data/Subsets/DAT_Representation/scan_processed.dat'


def main():
    # Set to true to see individual cells after segmentation for
    DEBUG = False

    DEVANAGARI = 'Devanagari_'
    WESTERN_A = 'Western_Arabic_'

    # initials = ['tz', 'ar', 'pr', 'rr', 'jh', 'lm', 'bk']
    initials = ['tz', 'ar', 'pr', 'jh', 'lm', 'bk']

    master_outs = []

    for init in initials:

        D = 'Data/scan_jpg/' + DEVANAGARI + init + '.png'
        W = 'Data/scan_jpg/' + WESTERN_A + init + '.png'
        files = [D, W]
        for i in range(len(files)):
            if i == 0:
                L = 'D'
            elif i == 1:
                L = 'W'
            else:
                exit(-1)


            '''
            Reads an image from file name and converts it to pseudobinary
            '''
            im_th = im2Binary(files[i])


            im_eroded = erode(im_th, 2)

            '''
            Floodfills the grid of the image, leaving only the unconnected written numbers
            '''
            im_floodfill = floodGrid(im_eroded)

            ''' 
            Segments the image into 10x14 matrix stored as 40px^2 cv2 Image Objects in form:
                    seg_mat = [
                                        col #
                    row #       0   1   2   3   ... 9
                        0       -   -   -   -   -   -
                        1           ex
                        2
                        3
                        ...
                        13    
                    ]
                    where ex = seg_mat[1][1] = the second sample image (40x40) in the second column (written number 1)
            '''
            seg_mat = segment(im_floodfill)
            if len(seg_mat) == 10 and len(seg_mat[0]) == 14:
                pass
            else:
                raise AttributeError('Failure During Image Processing')

            for c in range(len(seg_mat)):
                for r in range(len(seg_mat[c])):
                    seg_mat[c][r] = center(seg_mat[c][r])
                    seg_mat[c][r] = resize(seg_mat[c][r])

            if len(seg_mat[0][0]) == 28:
                pass
            else:
                raise AttributeError('Failure During Image resizing/centering')

            data_out = formatToDat(seg_mat, L)

            if DEBUG:
                for i in seg_mat:
                    for j in i:
                        cv2.imshow("single values", j)
                        cv2.waitKey(0)
                cv2.destroyAllWindows()

            master_outs.append(data_out)
        print('Completed: ', init)

    dump = [[],[]]
    for data_out in master_outs:
        for i in range(len(data_out[0])):
            dump[0].append(data_out[0][i])
            dump[1].append(data_out[1][i])
    pickle.dump(dump, open(DAT_FILENAME, 'wb+'))

def formatToDat(seg_mat, L):
    subset = []
    labels = []
    # symbol
    for c in range(len(seg_mat)):
        #sample number
        for r in range(len(seg_mat[c])):
            subset.append(seg_mat[c][r])
            labels.append([L, c])
    data_out = [subset, labels]
    return data_out


def center(img):
    temp = copy.copy(erode(img, 3))
    temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
    d = temp
    s = np.shape(d)

    s2 = (s[0], s[1])

    # print(s, s2)
    rs = d[:, :, 0]

    r = rs.flatten()

    white_ind = np.all(d == 255, 2)
    # white_n = np.sum(np.sum(white_ind))
    nw_ind = np.logical_not(white_ind)
    nw_n = np.sum(np.sum(nw_ind))

    x = np.arange(0, s[1], 1)
    y = np.arange(0, s[0], 1)

    xx, yy = np.meshgrid(x, y)
    xm = np.sum(np.sum(xx[nw_ind])) / nw_n
    ym = np.sum(np.sum(yy[nw_ind])) / nw_n
    xd = xm - s[1] / 2
    yd = ym - s[0] / 2
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
    return d2


def resize(img):
    dim = (28, 28)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def erode(img, n):
    kernel = np.ones((n, n), np.uint8)
    img_dilation = cv2.erode(img, kernel, iterations=1)
    return img_dilation


def im2Binary(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    th, im_th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    return im_th


def floodGrid(im_th):
    im_floodfill = im_th.copy()
    seed = find_first_black(im_th)
    cv2.floodFill(im_floodfill, None, seed, 255)
    return im_floodfill


def find_first_black(img):
    h, w = img.shape[:2]
    for x in range(h):
        for y in range(w):
            '''
            An explanation of the for loop below:
                    This loop makes it more likely that the first row 
                    will get floodfilled in order from left to right, and
                    thus makes it very likely if not certain that the column
                    numbers will be accurate
            '''
            for l in range(50):
                if img[x + l, y] == 0:
                    return y, x + l
    raise ValueError('No black pixels found!')

def segment(im_floodfill):
    col_idx = [(0, 0) for i in range(10)]
    cols = []
    indiv = []

    h, w = im_floodfill.shape[:2]
    for i in range(10):
        seed = find_first_black(im_floodfill)
        if i == 0 and seed[0] - 15 < 0:
            col_idx[i] = (0, 40)
        else:
            col_idx[i] = (seed[0] - 15, seed[0] + 25)
        cv2.floodFill(im_floodfill, None, seed, 255)
        rowcrop = im_floodfill[0:h, col_idx[i][0]:col_idx[i][1]]
        cols.append(rowcrop)

    for i in range(len(cols)):
        cropped = []
        r = 0
        while r < h:
            for c in range(40):
                if cols[i][r, c] == 0:
                    crop = copy.copy(cols[i][r - 15: r + 25, 0:])
                    cv2.floodFill(cols[i], None, (c, r), 255)
                    if len(cropped) < 14:
                        cropped.append(crop)
                    r += 20
                    break
            r += 1
        indiv.append(cropped)
    return indiv

if __name__ == "__main__":
    main()