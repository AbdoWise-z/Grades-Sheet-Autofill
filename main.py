import csv
import os
import random
import time
import uuid

import cv2
import numpy as np
import skimage.io as io
from skimage.draw import rectangle
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.measure import find_contours
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import pandas as pd

digits_classifier = svm.LinearSVC(dual="auto")
path_to_dataset = r'digits_dataset'
path_to_inputs = r'inputs'
target_img_size = (32, 32)
svm_loaded = False


def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def load_dataset():
    features = []
    labels = []
    img_filenames = os.listdir(path_to_dataset)
    i = 0
    for fn in img_filenames:
        path = os.path.join(path_to_dataset, fn)
        for ffn in os.listdir(path):
            labels.append(fn)
            img_path = os.path.join(path, ffn)
            img = cv2.imread(img_path)
            features.append(extract_hog_features(img))
            # print(f"Loading : {img_path}")

            i += 1
            if i > 0 and i % 100 == 0:
                print("[INFO] processed {}".format(i))
    return features, labels


def load_svm():
    features, labels = load_dataset()
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2)
    digits_classifier.fit(train_features, train_labels)
    accuracy = digits_classifier.score(test_features, test_labels)
    print(f"accuracy: {accuracy * 100} %")
    global svm_loaded
    svm_loaded = True


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


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    plt.bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
    plt.show()


def getPaper(img) -> np.array:
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)  # remove noise
    _, edges = cv2.threshold(blurred_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(edges, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    hull = cv2.convexHull(approx_polygon)

    # cv2.drawContours(img, largest_contour, contourIdx=-1, thickness=15, color=(255))

    x, y, w, h = cv2.boundingRect(approx_polygon)
    src_points = np.float32(hull)

    if (len(src_points)) > 4:
        print("Can't detect the box")
        return None

    top_left = [0, 0]
    bot_left = [0, h]
    top_right = [w, 0]
    bot_right = [w, h]

    top_left_i = 0
    bot_left_i = 0
    top_right_i = 0
    bot_right_i = 0

    i = 0
    center = [0, 0]
    while i < len(src_points):
        center += src_points[i][0]
        i = i + 1

    center[0] = center[0] / 4
    center[1] = center[1] / 4
    i = 0
    while i < len(src_points):
        v = src_points[i][0] - center
        if v[0] < 0 and v[1] < 0:
            top_left_i = i
        if v[0] > 0 > v[1]:
            top_right_i = i
        if v[0] > 0 and v[1] > 0:
            bot_right_i = i
        if v[0] < 0 < v[1]:
            bot_left_i = i
        i = i + 1

    dst_points = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0],
    ])

    dst_points[bot_left_i] = bot_left
    dst_points[bot_right_i] = bot_right
    dst_points[top_left_i] = top_left
    dst_points[top_right_i] = top_right

    # print(src_points)
    # print(dst_points)

    # print(src_points.shape)
    # print(dst_points.shape)

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, perspective_matrix, (w, h))
    # show_images([warped_image])

    return warped_image


def getTable(img) -> np.array:
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)  # remove noise
    edges = cv2.Canny(blurred_image, 20, 120) # this is a really low threshold, but it works ..
    edges = cv2.dilate(edges, np.ones([3,3]))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    hull = cv2.convexHull(approx_polygon)

    # cv2.drawContours(img, largest_contour, contourIdx=-1, thickness=15, color=(255))
    # show_images([img , edges])

    x, y, w, h = cv2.boundingRect(hull)
    src_points = np.float32(hull)

    if (len(src_points)) > 4:
        print("Can't detect the box")
        return None

    top_left = [0, 0]
    bot_left = [0, h]
    top_right = [w, 0]
    bot_right = [w, h]

    top_left_i = 0
    bot_left_i = 0
    top_right_i = 0
    bot_right_i = 0

    i = 0
    center = [0, 0]
    while i < len(src_points):
        center += src_points[i][0]
        i = i + 1

    center[0] = center[0] / 4
    center[1] = center[1] / 4
    i = 0
    while i < len(src_points):
        v = src_points[i][0] - center
        if v[0] < 0 and v[1] < 0:
            top_left_i = i
        if v[0] > 0 > v[1]:
            top_right_i = i
        if v[0] > 0 and v[1] > 0:
            bot_right_i = i
        if v[0] < 0 < v[1]:
            bot_left_i = i
        i = i + 1

    dst_points = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0],
    ])

    dst_points[bot_left_i] = bot_left
    dst_points[bot_right_i] = bot_right
    dst_points[top_left_i] = top_left
    dst_points[top_right_i] = top_right

    # print(src_points)
    # print(dst_points)

    # print(src_points.shape)
    # print(dst_points.shape)

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, perspective_matrix, (w, h))
    # show_images([warped_image])

    return warped_image


def find_vertical_lines(edges, thickness=8, forward=7, value=255, accumulate=False) -> []:
    lines = []
    x = 0
    # print(edges.shape)
    last_can_add = True
    while x < edges.shape[1]:
        ok = True
        disc = forward
        for y in range(edges.shape[0]):
            _connected = False
            for t in range(thickness):
                _x = x + t - thickness // 2
                if _x < 0 or _x >= edges.shape[1]:
                    continue
                _connected = _connected or edges[y, _x] == value

            if not _connected:
                disc -= 1
                if disc == 0:
                    ok = False
                    break
            else:
                if not accumulate:
                    disc = forward
        if ok:
            if not last_can_add:
                lines.append(x)
                last_can_add = True
        else:
            if last_can_add:
                lines.append(x - thickness // 2 if x > thickness // 2 - 1 else 0)
                last_can_add = False
        x += 1
    if len(lines) == 1:
        return []
    return lines


def find_horizontal_lines(edges, thickness=9, forward=7, value=255, accumulate=False) -> []:
    lines = []
    y = 0
    # print(edges.shape)
    last_can_add = True
    while y < edges.shape[0]:
        ok = True
        disc = forward
        for x in range(edges.shape[1]):
            _connected = False
            for t in range(thickness):
                _y = y + t - thickness // 2
                if _y < 0 or _y >= edges.shape[0]:
                    continue
                _connected = _connected or edges[_y, x] == value

            if not _connected:
                disc -= 1
                if disc == 0:
                    ok = False
                    break
            else:
                if not accumulate:
                    disc = forward
        if ok:
            if not last_can_add:  # this is the first line
                lines.append(y)
                last_can_add = True
        else:
            if last_can_add:
                lines.append(y - thickness // 2 if y > thickness // 2 - 1 else 0)
                last_can_add = False
        y += 1

    if len(lines) == 1:
        return []
    return lines


def extract_table(table):
    # img = table.copy()
    blurred_image = cv2.GaussianBlur(table, (5, 5), 0)  # remove noise
    _, edges = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # edges = cv2.Canny(blurred_image, 50, 150)
    # edges = cv2.dilate(edges, np.ones([5, 5]))
    # show_images([edges])
    cv2.imwrite("out.png", edges)

    # TODO: performance heavy (I know there is a better way to do it but meh)
    ver = find_vertical_lines(edges,
                              forward=int(edges.shape[0] * 0.1),
                              thickness=int(edges.shape[1] * 0.01),
                              accumulate=True)
    hor = find_horizontal_lines(edges, forward=int(edges.shape[1] * 0.1),  accumulate=True)

    print(ver[1] - ver[0])

    # cv2.imwrite("out.png", edges)

    # now calculate all the boxes we need
    i = 1
    codes = []  # x1 y1 x2 y1
    quests = []  #
    # codes_images = []
    # questions_images = []

    # print(table.shape)
    while i < len(hor) // 2:
        codes.append([ver[0], hor[i * 2], ver[1], hor[i * 2 + 1]])
        # codes_images.append(table[hor[i]: hor[i + 1], ver[0]: ver[1]])
        q = []
        # qi = []
        for k in range(3, len(ver) // 2):
            q.append([ver[k * 2], hor[i * 2], ver[k * 2 + 1], hor[i * 2 + 1]])
            # qi.append(table[hor[i]: hor[i + 1], ver[k]: ver[k + 1]])
        quests.append(q)
        # questions_images.append(qi)
        i = i + 1

    # debug stuff
    # i = 0
    # for img in codes_images:
    #     cv2.imwrite(f"code{i}.png", img)
    #     i = i + 1
    # show_images([edges, table, img])

    return codes, quests


def apply_padding(items, Xfactor, Yfactor, Xtop=0.0, Ytop=0.0):
    out = []
    for k in items:
        w = k[2] - k[0]
        h = k[3] - k[1]
        xT = int(Xtop * w)
        yT = int(Ytop * h)
        xO = int(Xfactor * w)
        yO = int(Yfactor * h)
        out.append([k[0] + xO + xT, k[1] + yO + yT, k[2] - xO, k[3] - yO])
    return out


digit_padding = 4


def toDigit(digit_image):
    # leave a "digit_padding" px in evey dimension
    scale = (target_img_size[1] - digit_padding * 2) / digit_image.shape[0]

    # digit_image = cv2.bitwise_not(digit_image)
    # digit_image = cv2.threshold(digit_image, 50, 255, cv2.THRESH_TOZERO)

    # print(scale)
    digit_image = cv2.resize(digit_image, None, fx=scale, fy=scale)
    # showHist(digit_image)
    # show_images([digit_image])
    # print(digit_image.shape)

    img = np.zeros(target_img_size)
    xO = digit_image.shape[1] / digit_image.shape[0]  # wid / hei
    xO = (1 - xO) / 2
    xO = round(target_img_size[0] * xO)
    img[digit_padding:target_img_size[0] - digit_padding, xO:digit_image.shape[1] + xO] = digit_image
    img = img.astype(np.uint8)

    # # print(digit_image.shape)
    # # print(xO)
    #
    # img[digit_padding:target_img_size[0] - digit_padding, xO:digit_image.shape[1] + xO] = digit_image
    # img = img.astype(np.uint8)

    # cv2.imwrite(f"digits_dataset/{str(uuid.uuid4())}.jpg", img)
    # cv2.imwrite(f"digits_dataset/{int(time.time() * 1000000)}-d.jpg", digit_image)

    # io.imshow(img)
    # io.show()

    if not svm_loaded:
        return 'X'

    features = extract_hog_features(img)
    predict = digits_classifier.predict([features])[0]
    # print(predict)
    return f'{predict}'


def split_code_image(code_image, peak_offset=0.4) -> []:
    # code_image = cv2.Laplacian(code_image, 5)
    # print("sharpen")
    hist = cv2.calcHist([code_image], [0], None, [256], [0, 256])
    peak = 0
    for i in range(len(hist)):
        if hist[i] >= hist[peak]:
            peak = i

    code_image = gamma_correction(code_image, 0.2)
    # showHist(code_image)
    _, thresholdImage = cv2.threshold(code_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresholdImage = cv2.GaussianBlur(thresholdImage, (3, 3), 0)
    thresholdImage = cv2.erode(thresholdImage, np.ones([2,2]))

    show_images([thresholdImage])
    # thresholdImage = cv2.erode(thresholdImage, np.ones([2,2]))

    ver = find_vertical_lines(thresholdImage, value=0, forward=1, thickness=1)
    hor = find_horizontal_lines(thresholdImage, value=0, forward=3, thickness=1)

    if len(hor) != 2:   # this is not a student code
        return [], []


    codes = []
    codes_seg = []

    i = 0
    while i < len(ver) // 2:
        codes.append(code_image[hor[0]:hor[1], ver[i * 2]:ver[i * 2 + 1]])
        codes_seg.append(thresholdImage[hor[0]:hor[1], ver[i * 2]:ver[i * 2 + 1]])
        i += 1

    if len(codes) != 7:
        # print("Failed to extract code from the image ..")
        # before failing. try to fix this miss
        avg = 0
        for k in codes:
            avg += k.shape[1]
        avg /= 7   # not averaging by the len because it should be 7

        filtered_cd = []
        filtered_csd = []

        for i, k in enumerate(codes):
            if k.shape[1] > avg * 0.4:  # only add digits that's more than the 20% of the average value
                filtered_cd.append(k)   # others are probably some random pixels
                filtered_csd.append(codes_seg[i])
            else:
                print("Skipped a very small digit")

        codes = filtered_cd
        codes_seg = filtered_csd
        filtered_cd = []
        filtered_csd = []

        for i, k in enumerate(codes):
            k_csd = codes_seg[i]
            v = k.shape[1] / avg
            if v > 1.5:                  # we probably got two digits (or more) in the same image
                print("Attempting to split a wide digit image")
                split = round(v)
                for j in range(split):
                    filtered_cd .append(k[:, k.shape[1] // split * j: k.shape[1] // split * (j+1)])
                    filtered_csd.append(k_csd[:, k_csd.shape[1] // split * j: k_csd.shape[1] // split * (j + 1)])
                    # show_images([k, k_csd])

            else:
                filtered_cd.append(k)
                filtered_csd.append(k_csd)

        codes = filtered_cd
        codes_seg = filtered_csd
        filtered_cd = []
        filtered_csd = []

        if len(codes) != 7:   # ok I give up
            print("failed to read digits")
            show_images([code_image, thresholdImage])
            return [], []

    return codes, codes_seg


def resolve_student(gray_image, code_location, answers_locations) -> [[]]:
    # blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # remove noise
    code_image = gray_image[code_location[1]:code_location[3], code_location[0]:code_location[2]]
    # apply a threshold to convert to binary image

    code_digits, code_seg_digit = split_code_image(code_image)

    if len(code_digits) != 7:
        print("Skipping bad digits ..")
        return

    code = ''
    for k in code_seg_digit:
        code += toDigit(k)

    print(code)
    pass


def gamma_correction(img, gamma=1.0):
    if gamma == 1:
        return img

    img = img.astype(np.float32) / 255
    img = np.power(img, gamma)
    return (img * 255).astype(np.uint8)


def ProcessImage(file):
    print("Loading Image ..")
    img = cv2.imread(file)
    print(f"Image Loaded : {file}")
    print("Applying Image Pre-processing ...")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = gamma_correction(gray_image)

    # gray_image = cv2.equalizeHist(gray_image)
    # show_images([gray_image])
    # cv2.imwrite("out.png", gray_image)
    # first we define the page borders
    # we know that a page is generally while

    print("Detecting Table ..")
    paper = getPaper(gray_image)
    if paper is None:
        return
    table = getTable(paper)
    if table is None:
        return
    show_images([paper, table])
    print("Extracting cells ..")
    # extracting codes / answers and applying some padding
    codes_positions, questions_positions = extract_table(table)
    codes_positions = apply_padding(codes_positions, Xfactor=0.05, Yfactor=0.05, Xtop=0.00, Ytop=0.0)
    print(f"Extracted a total of : {len(codes_positions)} rows , Qs: {len(questions_positions[0])}")  # not safe
    qp = []
    for k in questions_positions:
        qp.append(apply_padding(k, Xfactor=0.00, Yfactor=0.15))
    questions_positions = qp

    print("Converting images ..")
    students = []
    for i in range(len(codes_positions)):
        students.append(resolve_student(table, codes_positions[i], questions_positions[i]))


def main():
    print("Loading SVM...")
    load_svm()
    print("SVM Loaded")

    print("Starting ..")
    for fn in os.listdir(path_to_inputs):
        ProcessImage(os.path.join(path_to_inputs, fn))

    print("Ok done :)")


if __name__ == "__main__":
    main()
