from PIL import Image
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.image as img
import sounddevice as sd
import soundfile as sf
import numpy as np
import cv2
from operator import add
from functools import reduce
from tqdm import tqdm
import sys


def plot(img1, img2):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title('orginalny obraz')
    ax2 = fig.add_subplot(212)
    ax2.set_title('kompresja')
    # ax1.suptitle('orginał ', fontsize=16)
    ax1.imshow(img1)
    ax2.imshow(img2.astype(int))#as type int sotosowac w zaleznosci od obrazka doc4 nie dziala z
    plt.show()


def RLE_encode(img):
    y, x = img.shape[:2]
    size = np.array([y, x])

    img2 = img.copy()
    # img2[0,0,0]=1#flatten dziala po 0,1,2 y? wszytkie r wartosci po ois oy najpeir potem g potem b
    # img2[1,0,0]=2
    # img2[2,0,0]=3
    # imgf=img2.flatten()
    minus_one_dim = img2.reshape(len(img2), -1)
    imgf = minus_one_dim.flatten(order='F')

    # print("rozmiar ", y, "  ", x)
    # print('flatten: ', imgf)
    # print('flatten2: ', len(imgf))

    length = imgf.size
    ctr = 1
    RLE = np.array([])

    for i in tqdm( range(0, length - 1)):

        if (imgf[i] == imgf[i + 1]):
            ctr = ctr + 1
            if (i == length - 2):
                RLE = np.append(RLE, ctr)
                RLE = np.append(RLE, imgf[i])

        else:
            # a = np.array([imgf[i-1],ctr])
            # print(a)
            # RLE = np.append(RLE,a, axis=0)
            RLE = np.append(RLE, ctr)
            RLE = np.append(RLE, imgf[i])
            ctr = 1
            # print(RLE)

    # print('rozm przed', len(RLE))
    RLE = np.append(RLE, y)
    RLE = np.append(RLE, x)
    # print('rozm po', len(RLE))

    return RLE


def RLE_decode2(rle):
    out_array = []

    x = int(rle[len(rle) - 1])
    y = int(rle[len(rle) - 2])

    # print('y i x :', y, x)

    rle = np.delete(rle, len(rle) - 1)
    rle = np.delete(rle, len(rle) - 1)
    # print('rozm po', len(rle))
    # print('y,x :',y,x)
    # print(rle)

    for i in tqdm(range(0, len(rle), 2)):
        out_array.extend([rle[i + 1] for j in range(int(rle[i]))])

    # print('rozmiar out array:', len(out_array))
    out_array = np.array(out_array)

    decoded_t = out_array.reshape(x * 3, -1).T
    final_decoded = decoded_t.reshape(y, x, 3)

    return final_decoded

    # print('test',RLE[2])
    # print(imgf.size)
    # print(imgf.shape[1])

    # np.vstack((size,imgf))
    # imgf=np.concatenate([size,imgf])
    # print('flatten3: ',imgf)#rozmiar na początku y,x


def split4(image):  # jak slplit
    half_split = np.array_split(image, 2)  # ,array split dzieli na 4 nawet gdy nieparzyste
    res = map(lambda x: np.array_split(x, 2, axis=1),
              half_split)  # map(operacja,przedmniot operacji(dla kazdego elementu przedmiotu))
    return reduce(add, res)  # co robi add i reduce???


class leaf:
    def __init__(self, P1, P4, Color_val):
        self.p1 = P1  # wsp rogow fragmentu obrazu
        self.p4 = P4
        self.color_val = Color_val


leafList = []
p1List = np.array([])
p4List = np.array([])
color_valList = np.array([])


class QuadTree:
    # def __init__(self):

    def loop(self, img, P1, img2):
        # print('img  macierz wyglada tak')
        # print(img)
        # print('------------')
        split_img = split4(img)
        # print("rozmiar splita: ", len(split_img))

        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(split_img[0])
        # axs[0, 1].imshow(split_img[1])
        # axs[1, 0].imshow(split_img[2])
        # axs[1, 1].imshow(split_img[3])
        # plt.show()

        # if (len(split_img) != 4):
        #     return 0

        # NwY, NwX = split_img[0].shape[:2]
        # NeY, NeX = split_img[1].shape[:2]
        # SwY, SwX = split_img[2].shape[:2]
        # SeY, SeX = split_img[3].shape[:2]

        # print("------Rozmiary podziału-----------")
        # print("Nw:", NwY, "----", NwX)
        # print("Ne:", NeY, "----", NeX)
        # print("Sw:", SwY, "----", SwX)
        # print("Se:", SeY, "----", SeX)

        P1Nw = np.array([P1[0], P1[1]])
        P1Ne = np.array([P1Nw[0], P1Nw[1] + split_img[0].shape[1]])
        P1Sw = np.array([P1Nw[0] + split_img[0].shape[0], P1Nw[1]])
        P1Se = np.array([P1Nw[0] + split_img[0].shape[0], P1Nw[1] + split_img[0].shape[1]])

        P4Nw = np.array([split_img[0].shape[0] - 1 + P1Nw[0], split_img[0].shape[1] - 1 + P1Nw[1]])
        P4Ne = np.array([split_img[1].shape[0] - 1 + P1Ne[0], split_img[1].shape[1] - 1 + P1Ne[1]])
        P4Sw = np.array([split_img[2].shape[0] - 1 + P1Sw[0], split_img[2].shape[1] - 1 + P1Sw[1]])
        P4Se = np.array([split_img[3].shape[0] - 1 + P1Se[0], split_img[3].shape[1] - 1 + P1Se[1]])

        # P1Nw = P1
        # # P2Nw = [P1[0], split_img[0].shape[1] - 1]  # po x to shape[1]-1 za indeksowanie od 0
        # P3Nw = [split_img[0].shape[0] - 1, P1[1]]
        # P4Nw = [split_img[0].shape[0] - 1, split_img[0].shape[1] - 1]
        #
        # P1Ne = [P1[0], split_img[0].shape[1]]  # bez -1 bo o jeden piksel dalej niz cwiartka Nw
        # # P2Ne = [P1[0], split_img[0].shape[1] + split_img[1].shape[1] - 1]
        # # P3Ne = [split_img[1].shape[0] - 1, split_img[0].shape[1]]
        # P4Ne = [split_img[1].shape[0] - 1, split_img[0].shape[1] + split_img[1].shape[1] - 1]
        #
        # P1Sw = [P3Nw[0] + 1, P3Nw[1]]
        # # P2Sw = [P4Nw[0] + 1, P4Nw[1]]
        # # P3Sw = [split_img[0].shape[0] + split_img[2].shape[0] - 1, P1[1]]
        # P4Sw = [split_img[0].shape[0] + split_img[2].shape[0] - 1, split_img[0].shape[1] - 1]
        #
        # P1Se = [split_img[0].shape[0], split_img[0].shape[1]]
        # # P2Se = [split_img[1].shape[0], split_img[0].shape[1] + split_img[1].shape[1] - 1]
        # # P3Se = [split_img[0].shape[0] + split_img[2].shape[0] - 1, split_img[0].shape[1]]
        # P4Se = [split_img[0].shape[0] + split_img[2].shape[0] - 1, split_img[0].shape[1] + split_img[1].shape[1] - 1]

        # print("------test wspolrzednych-----------")
        # print("P1Nw: ", P1Nw[0], '-', P1Nw[1])
        # print("P2Nw: ", P2Nw[0], '-', P2Nw[1])
        # print("P3Nw: ", P3Nw[0], '-', P3Nw[1])
        # print("P4Nw: ", P4Nw[0], '-', P4Nw[1])
        # print()
        # print("P1Ne: ", P1Ne[0], '-', P1Ne[1])
        # print("P2Ne: ", P2Ne[0], '-', P2Ne[1])
        # print("P3Ne: ", P3Ne[0], '-', P3Ne[1])
        # print("P4Ne: ", P4Ne[0], '-', P4Ne[1])
        # print()
        # print("P1Sw: ", P1Sw[0], '-', P1Sw[1])
        # print("P2Sw: ", P2Sw[0], '-', P2Sw[1])
        # print("P3Sw: ", P3Sw[0], '-', P3Sw[1])
        # print("P4Sw: ", P4Sw[0], '-', P4Sw[1])
        # print()
        # print("P1Se: ", P1Se[0], '-', P1Se[1])
        # print("P2Se: ", P2Se[0], '-', P2Se[1])
        # print("P3Se: ", P3Se[0], '-', P3Se[1])
        # print("P4Se: ", P4Se[0], '-', P4Se[1])

        sameColor1 = 1
        sameColor2 = 1
        sameColor3 = 1
        sameColor4 = 1
        find = 0

        if split_img[0].size != 0:
            for i in range(0, split_img[0].shape[0]):  # i =y
                for j in range(0, split_img[0].shape[1]):  # j =x
                    if (split_img[0][0, 0, 0] != split_img[0][i, j, 0] or
                            split_img[0][0, 0, 1] != split_img[0][i, j, 1] or
                            split_img[0][0, 0, 2] != split_img[0][i, j, 2]):
                        sameColor1 = 0
                        Q1 = QuadTree()
                        Q1.loop(split_img[0], P1Nw, img2)
                        find = 1
                        break
                if find == 1:
                    break
            find = 0
            # if()
            if sameColor1 == 1:
                # pbar.update(1)
                # print('wsp koloru p1:',P1Nw[0],P1Nw[1])
                # print('wsp koloru p4:', P4Nw[0], P4Nw[1])
                leafList.append(leaf(P1Nw, P4Nw, split_img[0][0, 0, :]))
                # newImg = self.reconstruct(img2)
                # plot(img1, newImg)
                # plt.show()

        if split_img[1].size != 0:
            for i in range(0, split_img[1].shape[0]):  # i =y
                for j in range(0, split_img[1].shape[1]):  # j =x
                    if (split_img[1][0, 0, 0] != split_img[1][i, j, 0] or
                            split_img[1][0, 0, 1] != split_img[1][i, j, 1] or
                            split_img[1][0, 0, 2] != split_img[1][i, j, 2]):
                        sameColor2 = 0
                        Q2 = QuadTree()
                        Q2.loop(split_img[1], P1Ne, img2)
                        find = 1
                        break
                if find == 1:
                    break
            find = 0

            if sameColor2 == 1:
                # pbar.update(1)
                leafList.append(leaf(P1Ne, P4Ne, split_img[1][0, 0, :]))
                # newImg = self.reconstruct(img2)
                # plot(img1, newImg)
                # plt.show()

        if split_img[2].size != 0:
            for i in range(0, split_img[2].shape[0]):  # i =y
                for j in range(0, split_img[2].shape[1]):  # j =x
                    if (split_img[2][0, 0, 0] != split_img[2][i, j, 0] or
                            split_img[2][0, 0, 1] != split_img[2][i, j, 1] or
                            split_img[2][0, 0, 2] != split_img[2][i, j, 2]):
                        sameColor3 = 0
                        Q3 = QuadTree()
                        Q3.loop(split_img[2], P1Sw, img2)
                        find = 1
                        break
                if find == 1:
                    break
            find = 0

            if sameColor3 == 1:
                # pbar.update(1)
                leafList.append(leaf(P1Sw, P4Sw, split_img[2][0, 0, :]))
                # newImg = self.reconstruct(img2)
                # plot(img1, newImg)
                # plt.show()

        if split_img[3].size != 0:
            for i in range(0, split_img[3].shape[0]):  # i =y
                for j in range(0, split_img[3].shape[1]):  # j =x
                    if (split_img[3][0, 0, 0] != split_img[3][i, j, 0] or
                            split_img[3][0, 0, 1] != split_img[3][i, j, 1] or
                            split_img[3][0, 0, 2] != split_img[3][i, j, 2]):
                        sameColor4 = 0
                        Q4 = QuadTree()
                        Q4.loop(split_img[3], P1Se, img2)
                        find = 1
                        break
                if find == 1:
                    break
            find = 0

            if sameColor4 == 1:
                # pbar.update(1)
                leafList.append(leaf(P1Se, P4Se, split_img[3][0, 0, :]))
                # newImg = self.reconstruct(img2)
                # plot(img1, newImg)
                # plt.show()

    def start(self, img, img2):
        y, x = img.shape[:2]
        P1 = [0, 0]
        leafList.append(leaf(int(y), int(x), 0))#pierwszy element listy to rozmiar obrazu
        self.loop(img, P1, img2)

        # split_img = split4(img)
        # NwY, NWX = split_img[0].shape[:2]
        # NwP1=[P1[]]

    def reconstruct(self, img):

        y = leafList[0].p1
        x = leafList[0].p4
        newImage = np.zeros([y, x, 3])
        # print('dlugosc', len(leafList))
        for i in range(1, len(leafList)):
            for j in range(leafList[i].p1[0], leafList[i].p4[0] + 1):
                for k in range(leafList[i].p1[1], leafList[i].p4[1] + 1):
                    # print('hmm',leafList[i].color_val)
                    newImage[j, k, :] = leafList[i].color_val

        return newImage


def show(img):
    RLE_coded = RLE_encode(img)

    rleDec = RLE_decode2(RLE_coded)
    plot(img, rleDec)
    #imgCompare(img1, rleDec)
    plt.show()

    # with tqdm(total=10000000000) as pbar:
    Q1 = QuadTree()
    Q1.start(img, img)

    QuadtreeDec = Q1.reconstruct(img)
    # print(leafList[0].p1, )
    plot(img, QuadtreeDec)
    #imgCompare(img1, QuadtreeDec)
    plt.show()

    print('rozmiar obrazu:', img.nbytes)
    print('rozmiar zakodowanego RLE:', RLE_coded.nbytes)
    quadtreesize = 0
    for i in range(1, len(leafList)):
        quadtreesize = leafList[i].p1.nbytes + quadtreesize
        quadtreesize = leafList[i].p4.nbytes + quadtreesize
        quadtreesize = quadtreesize + leafList[i].color_val.nbytes

    print('rozmiar zakodowanego quadtree:', quadtreesize)

def imgCompare(img1, img2):
    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            if img1[i, j, 0] != img2[i, j, 0] or img1[i, j, 1] != img2[i, j, 1] or img1[i, j, 1] != img2[i, j, 1]:
                print('obrazy nie sa takie same')


img1 = plt.imread('img7.jpg')
a = np.array([])

a = np.append(a, 1)
print('co jest', a)
show(img1)


