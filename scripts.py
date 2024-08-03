import os
import shutil

import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt


def writer():
    img_filenames = os.listdir(f'provided_dataset')
    char_to_number = {
        'a': "1",
        'b': "2",
        'c': "3",
        'd': "4",
        'e': "5",
        'f': "6",
        'g': "7",
        'h': "8",
        'i': "9",
    }

    for i, fn in enumerate(img_filenames):
        if fn.split('.')[-1] != 'jpg':
            continue

        label = char_to_number[fn.split('.')[0]]
        src = os.path.join("provided_dataset", fn)
        dst = os.path.join(os.path.join("hand_digits_dataset", label), fn)
        shutil.copyfile(src, dst)


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated
    # titles. images[0] will be drawn with the title titles[0] if exists You aren't required to understand this
    # function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def main():
    img = cv2.imread("inputs\\3.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("out.png" , img)
    show_images([img])


if __name__ == "__main__":
    main()
