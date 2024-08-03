import os

import cv2

DATA_SIZE = (100, 40)


def start():
    img_filenames = os.listdir(f'symbols')
    for i, fn in enumerate(img_filenames):
        print(f"Processing : {fn}")
        if not os.path.isdir(os.path.join('symbols', fn)):
            continue
        for i, f in enumerate(os.listdir(os.path.join('symbols', fn))):
            img = cv2.imread(os.path.join(os.path.join('symbols', fn), f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, DATA_SIZE)
            _, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow("img", img)
            cv2.imwrite(os.path.join("output", fn, f), img)


if __name__ == "__main__":
    start()
