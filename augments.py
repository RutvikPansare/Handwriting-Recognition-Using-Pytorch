import numpy as np
import PIL
import scipy.interpolate
from PIL import Image, ImageOps
import copy
import pickle
import cv2
from matplotlib import pyplot as plt

def im2nparr(im):
    return np.array(im)

def nparr2im(arr):
    m = np.max(np.max(np.max(arr)))
    if m > 255:
        arr_norm = arr/m
        arr_force_scale = 255 * arr_norm
    else:
        arr_force_scale = arr
    im = Image.fromarray(arr_force_scale)
    return im

def imrotRGB(im, theta=90):
    # Convert the image to a numpy array
    arr = im2nparr(im)

    # Get the individual R, G, and B channels
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Get the shape, this will be important later
    s = np.shape(arr)

    # Create some arbitrary arrays to represent the x and y grids
    # Need to center the images so that rotation matrix rotates around the origin
    x = np.arange(-s[0]/2, s[0]/2, 1)
    y = np.arange(-s[1]/2, s[1]/2, 1)
    xx, yy = np.meshgrid(x, y)

    # Flatten the 2D arrays to 1D for matrix multiplication
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    r_flat = r.flatten()
    g_flat = g.flatten()
    b_flat = b.flatten()

    # Compose the red, green, and blue matrices
    rmat = np.concatenate((xx_flat, yy_flat, r_flat), axis=1).transpose()
    gmat = np.concatenate((xx_flat, yy_flat, g_flat), axis=1).transpose()
    bmat = np.concatenate((xx_flat, yy_flat, b_flat), axis=1).transpose()

    # Compose the rotation matrix
    rotmat = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0,              0,             1]])

    # Create the rotated red, green, and blue matrices
    rmatr = (rmat * rotmat).transpose()
    gmatr = (gmat * rotmat).transpose()
    bmatr = (bmat * rotmat).transpose()

    # Resample the images using their original X and Y values
    rnew = scipy.interpolate.griddata((rmatr[:, 0], rmatr[:, 1]), rmatr[:, 2], (xx_flat, yy_flat))
    gnew = scipy.interpolate.griddata((gmatr[:, 0], gmatr[:, 1]), gmatr[:, 2], (xx_flat, yy_flat))
    bnew = scipy.interpolate.griddata((bmatr[:, 0], bmatr[:, 1]), bmatr[:, 2], (xx_flat, yy_flat))

    # Grab the R, G, and B columns
    rcol = rnew[2, :]
    gcol = gnew[2, :]
    bcol = bnew[2, :]

    # Reconfigure the shape of the matrices
    rfin = np.reshape(rcol, s)
    gfin = np.reshape(gcol, s)
    bfin = np.reshape(bcol, s)

    # Reassemble the matrices
    new_im_mat = np.concatenate((rfin, gfin, bfin), 2)

    # Convert the image matrix to an image
    new_im = nparr2im(new_im_mat)

    return new_im


def imrotBW(im, theta=90):
    # Convert the image to a numpy array
    arr = im2nparr(im)

    # Get the shape, this will be important later
    s = np.shape(arr)

    # Create some arbitrary arrays to represent the x and y grids
    # Need to center the images so that rotation matrix rotates around the origin
    x = np.arange(-s[0] / 2, s[0] / 2, 1)
    y = np.arange(-s[1] / 2, s[1] / 2, 1)
    xx, yy = np.meshgrid(x, y)

    # Flatten the 2D arrays to 1D for matrix multiplication
    xx_flat = np.reshape(xx, (-1, 1))
    yy_flat = np.reshape(yy, (-1, 1))
    z_flat = np.reshape(arr[:, :, 0], (-1, 1))

    # Compose the matrix
    zmat = np.concatenate((xx_flat, yy_flat, z_flat), axis=1).transpose()

    # Compose the rotation matrix
    rotmat = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])

    # Create the rotated black and white image
    zmatr = np.matmul(rotmat, zmat).transpose()

    zmatrx = zmatr[:, 0].flatten()
    zmatry = zmatr[:, 1].flatten()
    zmatrz = zmatr[:, 2].flatten()

    # Resample the images using their original X and Y values
    znew = scipy.interpolate.griddata((zmatrx, zmatry), zmatrz, (xx_flat, yy_flat), fill_value=np.nan)

    # # Fix the nan values
    znew[np.isnan(znew)] = 255
    # print(znew)

    # Grab the columns
    zcol = znew.astype(np.uint8)

    # Reconfigure the shape of the matrix
    zfin = np.reshape(zcol, (s[0], s[1])).astype(np.uint8)
    # print(zfin.shape)
    # Convert the image matrix to an image
    # new_im = np.zeros(s)
    # new_im[:, :, 0] = zfin
    # new_im[:, :, 1] = zfin
    # new_im[:, :, 2] = zfin

    return zfin

def threshold_image(image, thresh=127, open_image=False, filepath_dat='./image.dat', filepath_image='image.jpg'):
    if open_image:
        with Image.open(image) as img:
            mat = im2nparr(img)
            # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
            # np.mean(mat, 2)
            nm = np.round(np.mean(mat, 2), 0)
            # idp = nm >= thresh
            # idm = nm < thresh
            nm = (nm >= thresh) * 255 + (nm < thresh) * 0
            # nm[idm] = 0
            mat[:, :, 0] = copy.deepcopy(nm)
            mat[:, :, 1] = copy.deepcopy(nm)
            mat[:, :, 2] = copy.deepcopy(nm)
            print(np.shape(mat))
            # ret, thresh_out = cv2.threshold(mat, thresh, 255, cv2.THRESH_BINARY)
            thresh_im = nparr2im(mat)
            thresh_im.save(filepath_image, 'JPEG')
            file = open(filepath_dat, 'wb+')
            pickle.dump(mat, file)
    else:
        mat = im2nparr(image)
        ret, thresh_out = cv2.threshold(mat, thresh, 255, cv2.THRESH_BINARY)
        file = open(filepath_dat, 'wb+')
        pickle.dump(thresh_out, file)
    # plt.imshow(thresh_out, 'gray')
    # plt.show()

def skew_mat(im, theta=10, skewtype=0):
    # Convert the image to a numpy array
    arr = im2nparr(im)

    # Get the shape, this will be important later
    s = np.shape(arr)

    # Create some arbitrary arrays to represent the x and y grids
    # Need to center the images so that rotation matrix rotates around the origin
    x = np.arange(-s[0] / 2, s[0] / 2, 1)
    y = np.arange(-s[1] / 2, s[1] / 2, 1)
    xx, yy = np.meshgrid(x, y)

    # Flatten the 2D arrays to 1D for matrix multiplication
    xx_flat = np.reshape(xx, (-1, 1))
    yy_flat = np.reshape(yy, (-1, 1))
    z_flat = np.reshape(arr[:, :, 0], (-1, 1))

    # Compose the matrix
    zmat = np.concatenate((xx_flat, yy_flat, z_flat), axis=1).transpose()

    # Compose the skew matrix
    if skewtype==0:
        skewmat = np.array([[1, 0, 0],
                           [np.tan(theta), 1, 0],
                           [0, 0, 1]])
    elif skewtype==1:
        skewmat = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [np.tan(theta), 0, 1]])
    elif skewtype==2:
        skewmat = np.array([[1, np.tan(theta), 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    elif skewtype == 3:
        skewmat = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, np.tan(theta), 1]])
    elif skewtype == 4:
        skewmat = np.array([[1, 0, np.tan(theta)],
                            [0, 1, 0],
                            [0, 0, 1]])
    else:
        skewmat = np.array([[1, 0, 0],
                            [0, 1, np.tan(theta)],
                            [0, 0, 1]])


    # Create the rotated black and white image
    zmatr = np.matmul(skewmat, zmat).transpose()

    zmatrx = zmatr[:, 0].flatten()
    zmatry = zmatr[:, 1].flatten()
    zmatrz = zmatr[:, 2].flatten()

    # Resample the images using their original X and Y values
    znew = scipy.interpolate.griddata((zmatrx, zmatry), zmatrz, (xx_flat, yy_flat), fill_value=np.nan)

    # Fix the nan values
    znew[np.isnan(znew)] = 255

    # Grab the columns
    zcol = znew.astype(np.uint8)

    # Reconfigure the shape of the matrix
    zfin = np.reshape(zcol, (s[0], s[1])).astype(np.uint8)


    # Convert the image matrix to an image
    # new_im = np.zeros(s)
    # new_im[:, :, 0] = zfin
    # new_im[:, :, 1] = zfin
    # new_im[:, :, 2] = zfin

    return zfin


def bilinearGridInterp2D(x1y1, z1, x2y2, fill_value=0):
    # X1 = x1y1[0]
    # Y1 = x1y1[1]
    # s1 = np.shape(X1)
    # # x1min, x1max = X1[0, 0], X1[s1[0] - 1, s1[1] - 1]
    # # y1min, y1max = Y1[0, 0], Y1[s1[0] - 1, s1[1] - 1]
    #
    # X2 = x2y2[0]
    # Y2 = x2y2[1]
    # s2 = np.shape(X2)
    # # print(s1, s2, np.shape(z1))
    # # x2min, x2max = X2[0, 0], X2[s2[0] - 1, s2[1] - 1]
    # # y2min, y2max = Y2[0, 0], Y2[s2[0] - 1, s2[1] - 1]
    #
    # z2 = fill_value * np.ones((s2[0], s2[1]))
    #
    # for i in range(s1[1]-1):
    #     for j in range(s1[0]-1):
    #         xm = X1[i, j]
    #         xp = X1[i, j + 1]
    #         ym = Y1[i, j]
    #         yp = Y1[i + 1, j]
    #         # print((X2 >= xm))
    #         x2_ind = np.argwhere(np.logical_and((X2 >= xm), (X2 < xp)))
    #         y2_ind = np.argwhere(np.logical_and((Y2 >= ym), (Y2 < yp)))
    #         xs = np.shape(x2_ind)
    #         ys = np.shape(y2_ind)
    #         x2y2ind = np.empty((0, 2))
    #         for k in range(xs[0]):
    #             e = x2_ind[k, :]
    #             # print(e)
    #             for l in range(ys[0]):
    #                 if np.all(e == y2_ind[l, :]):
    #                     x2y2ind = np.append(x2y2ind, np.array([e]), axis=0)
    #         xys = np.shape(x2y2ind)
    #         # print(xs, ys)
    #         for k in range(xys[0]):
    #             # print(x2y2ind[k, 0], x2y2ind[k, 1])
    #             ix = x2y2ind[k, 0]
    #             iy = x2y2ind[k, 1]
    #             x = X2[ix.astype(int), iy.astype(int)]
    #             y = Y2[ix.astype(int), iy.astype(int)]
    #             x1 = xp
    #             x2 = xm
    #             y1 = yp
    #             y2 = ym
    #             C = 1/((x2-x1)*(y2-y1))
    #             XY = np.array([[x2*y2, -x2*y1, -x1*y2, x1*y1],
    #                              [ -y2,    y1,     y2,   -y1],
    #                              [ -x2,    x2,     x1,   -x1],
    #                              [ 1.0,  -1.0,   -1.0,   1.0]])
    #             F = np.array([[z1[i, j]],
    #                             [z1[i + 1, j]],
    #                             [z1[i, j + 1]],
    #                             [z1[i + 1, j + 1]]])
    #             A = C*XY*F
    #             a00 = A[0, 0]
    #             a10 = A[1, 0]
    #             a01 = A[2, 0]
    #             a11 = A[3, 0]
    #             z2[ix.astype(int), iy.astype(int)] = a00 + a10*x + a01*y + a11*x*y

    x = x1y1[0][0]
    y = x1y1[1][:, 0]
    z = z1
    xs = np.shape(x2y2[0])



    # x = [1, 2, 3, 4, 5]
    # y = [1, 2, 3, 4]
    #
    # z = np.ones((4,5))
    # z[0,:] = [11, 12, 13, 14, 15]
    # z[1,:] = [21, 22, 23, 24, 25]
    # z[2,:] = [31, 32, 33, 34, 35]
    # z[3,:] = [41, 42, 43, 44, 45]
    #
    # xp = 2.3
    # yp = 2.4

    # Check if x axis is monotonically increasing

    for i in range(len(x)):
        # Axis is not monotonically increasing
        # print((x[i] <= x[i - 1]))
        if (i > 1) and (x[i] <= x[i - 1]):
            # disp("Axis x is not monotonically increasing")
            break
    # Check if y axis is monotonically increasing
    for i in range(len(y)):
        # Axis is not monotonically increasing
        if i > 1 and y[i] <= y[i - 1]:
            # disp("Axis y is not monotonically increasing")
            break

    out = np.zeros(xs)

    for ix in range(xs[0]):
        for iy in range(xs[1]):
            xp = x2y2[0][ix, iy]
            yp = x2y2[1][ix, iy]
            z0 = False
            # Find x1 and x2 coordinates
            for i in range(len(x)):
                if (xp < x[0]) or xp > x[len(x)-1]:
                    z0 = True
                    break

                # Point x is the first point in the axis
                if xp == x[0]:
                    x1 = x[0]
                    x1_idx = 0
                    x2 = x[0]
                    x2_idx = 1
                    break

                # Point x is the last value in the axis
                if xp == x[len(x) - 1]:
                    x1 = x[len(x) - 2]
                    x1_idx = len(x) - 2
                    x2 = x[len(x) - 1]
                    x2_idx = len(x) - 1
                    break

                # Point x is in between first and last point of the axis
                if xp >= x[i] and xp <= x[i + 1]:
                    x1 = x[i]
                    x1_idx = i
                    x2 = x[i + 1]
                    x2_idx = i + 1
                    break

            # Find y1 and y2 coordinates
            for i in range(len(y)):
                # Point yp is outside range
                if np.logical_or(yp < y[0], yp > y[len(y) - 1]):
                    z0 = True
                    break

                # Point y is the first point in the axis
                if yp == y[0]:
                    y1 = y[0]
                    y1_idx = 0
                    y2 = y[1]
                    y2_idx = 1
                    break

                # Point y is the last value in the axis
                if yp == y[len(y) - 1]:
                    y1 = y[len(y) - 2]
                    y1_idx = len(y) - 2
                    y2 = y[len(y) - 1]
                    y2_idx = len(y) - 1
                    break

                # Point y is in between first and last point of the axis
                if yp >= y[i] and yp <= y[i + 1]:
                    y1 = y[i]
                    y1_idx = i
                    y2 = y[i + 1]
                    y2_idx = i + 1
                    break

            if z0:
                P = fill_value
            else:
                Q11 = z[y1_idx, x1_idx]
                Q12 = z[y2_idx, x1_idx]
                Q21 = z[y1_idx, x2_idx]
                Q22 = z[y2_idx, x2_idx]

                R1 = Q11 * ((x2 - xp) / (x2 - x1)) + Q21 * ((xp - x1) / (x2 - x1))
                R2 = Q12 * ((x2 - xp) / (x2 - x1)) + Q22 * ((xp - x1) / (x2 - x1))
                P = R1 * ((y2 - yp) / (y2 - y1)) + R2 * ((yp - y1) / (y2 - y1))

            out[ix, iy] = P
    return out

