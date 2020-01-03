__all__ = ['detect_scales_from_image']

import cv2
import numpy as np

import pytesseract as ocr

from sklearn.cluster import OPTICS

import time

import warnings
warnings.filterwarnings('ignore')

# import logging
#
# logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


ocr_config = '-l eng --oem 3 --psm 7'
max_fail = 5
min_success = 3


def preprocess(img):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarize image by thresholding
    # img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # filter out dense area
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    kernel = np.ones((3, 3))
    img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=10)
    img_close_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel, iterations=1)
    img_final = img_binary.copy()
    img_final[img_close_open == 255] = 0

    return img_binary, img_final


def endpoint_proposal(binarized_img, rho_interval=1, theta_interval=np.pi/2):
    # detect lines
    lines = cv2.HoughLines(binarized_img, rho_interval, theta_interval, threshold=150)
    if lines is None:
        return None

    lines = lines[:, 0, :]

    img_lines = np.zeros_like(binarized_img)
    for l in lines:
        rho, theta = l
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 100000 * (-b))
        y1 = int(y0 + 100000 * (a))
        x2 = int(x0 - 100000 * (-b))
        y2 = int(y0 - 100000 * (a))

        if abs(x1-x2) < 3:
            x1, x2 = (x1+x2)//2, (x1+x2)//2
        if abs(y1-y2) < 3:
            y1, y2 = (y1+y2)//2, (y1+y2)//2

        cv2.line(img_lines, (x1, y1), (x2, y2), 255, 1)

    # detect common cross points
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img_point = cv2.erode(img_lines, kernel)
    pts = np.where(img_point == 255)[::-1]
    pts = list(zip(*pts))
    return pts


def prepare_endpoint_boxes(pts, img_binary, box_size=7):
    h, w = img_binary.shape

    boxes = []
    points = []

    for p in pts:
        x1 = p[0] - box_size
        x2 = p[0] + box_size + 1
        y1 = p[1] - box_size
        y2 = p[1] + box_size + 1
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        box = img_binary[y1:y2, x1:x2]
        if np.count_nonzero(box == 255) == 0:
            continue

        kernel = np.ones((1, 15))
        box_hor = cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((15, 1))
        box_ver = cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel)
        box_erode = box.copy()
        box_erode[box_hor == 255] = 0
        box_erode[box_ver == 255] = 0

        if np.count_nonzero(box_erode == 255) == 0:
            continue

        boxes.append(box.reshape(-1))
        points.append(p)

    return np.array(points), np.array(boxes)


def read_number_using_tesseract(text_roi):
    text = ocr.image_to_string(text_roi, config=ocr_config)
    try:
        number = float(text)
        if number == 0:
            return False
        else:
            return number
    except ValueError:
        return False


def detect_scales_from_image(img_path):
    SEED = 123
    np.random.seed(SEED)

    start_time = time.time()

    img = cv2.imread(img_path)

    img_binary, img_final = preprocess(img)

    endpoints = endpoint_proposal(img_final)

    if not endpoints:
        return False

    endpoints, boxes = prepare_endpoint_boxes(endpoints, img_binary)

    if len(boxes) < 5:
        return False

    labels = OPTICS(metric='jaccard').fit_predict(boxes)

    cluster_labels = np.unique(labels)

    reading_success = {}
    reading_result = {}
    success_flag = False
    most_success = None

    for i, c in enumerate(cluster_labels):
        if c == -1:  # outliers
            continue

        cluster = endpoints[labels == c]
        unique_x = np.unique(cluster[:, 0])
        unique_y = np.unique(cluster[:, 1])

        num_fail = 0
        failure_flag = False

        # process horizontal lines
        for y in unique_y:
            hor_line = cluster[cluster[:, 1] == y]
            if len(hor_line) > 1:
                last_point = hor_line[0]
                for next_point in hor_line[1:]:
                    x1 = last_point[0]
                    x2 = next_point[0]
                    y2 = next_point[1]
                    y1 = y2 - 30

                    img_roi = img_binary[y1:y2, x1:x2]

                    kernel = np.ones((3, 8))
                    img_roi_dilate = cv2.dilate(img_roi, kernel)

                    contours, hierarchy = cv2.findContours(img_roi_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
                    bboxes = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in bboxes]

                    x_mid = (x2 - x1) // 2
                    y_mid = (y2 - y1) // 2

                    for box in bboxes:
                        if box[0] < x_mid < box[2] and box[1] < y_mid < box[3]:
                            x1, x2 = x1 + box[0], x1 + box[2]
                            y1, y2 = y1 + box[1], y1 + box[3]

                            text_roi = img[y1:y2, x1:x2]

                            if text_roi.shape[1] > 100 or text_roi.shape[1] < 5:
                                num_fail += 1
                                if num_fail == max_fail:
                                    failure_flag = True
                                break

                            # read_number_using_shape_matching(text_roi)
                            length = read_number_using_tesseract(text_roi)

                            if not length:
                                num_fail += 1
                                if num_fail == max_fail:
                                    failure_flag = True
                            else:
                                # print('Reading number:', length)
                                # print('Scale:', (next_point[0] - last_point[0])/length*1000)
                                ratio = (next_point[0] - last_point[0])/length*1000
                                scale = {'p1': tuple(last_point),
                                         'p2': tuple(next_point),
                                         'distance': float(next_point[0] - last_point[0]),
                                         'length': length}
                                if int(ratio) not in reading_success:
                                    reading_success[int(ratio)] = [ratio]
                                    reading_result[int(ratio)] = [scale]
                                else:
                                    reading_success[int(ratio)].append(ratio)
                                    reading_result[int(ratio)].append(scale)
                                    if len(reading_success[int(ratio)]) == min_success:
                                        success_flag = True
                                        most_success = int(ratio)
                            break

                    last_point = next_point
                    if failure_flag or success_flag:
                        break
            if failure_flag or success_flag:
                break
        if failure_flag or success_flag:
            continue

        # process vertical lines
        for x in unique_x:
            ver_line = cluster[cluster[:, 0] == x]
            if len(ver_line) > 1:
                last_point = ver_line[0]
                for next_point in ver_line[1:]:
                    y1 = last_point[1]
                    y2 = next_point[1]
                    x2 = last_point[0]
                    x1 = x2 - 30

                    img_roi = img_binary[y1:y2, x1:x2]
                    img_roi = cv2.rotate(img_roi, cv2.ROTATE_90_CLOCKWISE)

                    kernel = np.ones((3, 8))
                    img_roi_dilate = cv2.dilate(img_roi, kernel)

                    contours, hierarchy = cv2.findContours(img_roi_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
                    bboxes = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in bboxes]

                    h_roi, w_roi = img_roi.shape
                    x_mid = w_roi // 2
                    y_mid = h_roi // 2

                    for box in bboxes:
                        if box[0] < x_mid < box[2] and box[1] < y_mid < box[3]:
                            x1, x2 = x1 + box[1], x1 + box[3]
                            y1, y2 = y1 + w_roi - box[2], y1 + w_roi - box[0]

                            text_roi = img[y1:y2, x1:x2]

                            if text_roi.shape[1] > 100 or text_roi.shape[1] < 5:
                                num_fail += 1
                                if num_fail == max_fail:
                                    failure_flag = True
                                break

                            length = read_number_using_tesseract(text_roi)
                            if not length:
                                num_fail += 1
                                if num_fail == max_fail:
                                    failure_flag = True
                            else:
                                ratio = (next_point[1] - last_point[1]) / length * 1000
                                scale = {'p1': tuple(last_point),
                                         'p2': tuple(next_point),
                                         'distance': float(next_point[1] - last_point[1]),
                                         'length': length}
                                if int(ratio) not in reading_success:
                                    reading_success[int(ratio)] = [ratio]
                                    reading_result[int(ratio)] = [scale]
                                else:
                                    reading_success[int(ratio)].append(ratio)
                                    reading_result[int(ratio)].append(scale)
                                    if len(reading_success[int(ratio)]) == min_success:
                                        success_flag = True
                                        most_success = int(ratio)
                            break

                    last_point = next_point
                    if failure_flag or success_flag:
                        break
            if failure_flag or success_flag:
                break
        if failure_flag or success_flag:
            continue

    run_time = time.time() - start_time

    if most_success:
        if reading_result[most_success][0]['length'] > 1000:
            unit = 'mm'
        elif reading_result[most_success][0]['length'] < 100:
            unit = 'm'
        else:
            unit = 'mm'

        result = {'scales': reading_result[most_success],
                  'ratio': sum(reading_success[most_success]) / len(reading_success[most_success]),
                  'unit': unit,
                  'runtime': run_time
                  }
        return result
    else:
        for reading in reading_success:
            if len(reading_success[reading]) == 2:
                most_success = reading

                if reading_result[most_success][0]['length'] > 1000:
                    unit = 'mm'
                elif reading_result[most_success][0]['length'] < 100:
                    unit = 'm'
                else:
                    unit = 'mm'

                result = {'scales': reading_result[most_success],
                          'ratio': sum(reading_success[most_success]) / len(reading_success[most_success]),
                          'unit': unit,
                          'runtime': run_time
                          }
                return result
        else:
            return False


# if __name__ == '__main__':
#     r = detect_scales_from_image('../data/test_img.jpeg')
#     from pprint import pprint
#     pprint(r)
#     print(type(r['scales'][0]['p1'][0]))

    # img = cv2.imread('../test_img.jpeg')
    # for s in r['scales']:
    #     img = cv2.line(img, tuple(s[0]), tuple(s[1]), (0, 0, 255), 5)
    # cv2.imshow('', img)
    # cv2.waitKey()