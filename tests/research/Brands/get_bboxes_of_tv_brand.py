

import cv2
import argparse
import numpy
import os
from PIL import Image


def get_image_data(key):
    source_dtype = numpy.float32
    img = cv2.imread(key)
    if img is None:
        print("Unable to read %s" % key)
        return
    # return img.astype(source_dtype)
    return img


def get_bboxes_of_tv_brand(
        folder_name, dst_width, dst_height, tresh_min, tresh_max):
    for root_p, _tmp, files in os.walk(folder_name, followlinks=True):
        for file in files:
            if (os.path.splitext(file)[1] == ".JPG" or
                    os.path.splitext(file)[1] == ".jpg"):
                abs_path = os.path.join(root_p, file)
                image = get_image_data(abs_path)
                if image is None:
                    continue
                size = image.shape
                height, width, chans = size
                # data = cv2.resize(
                #     image, (dst_width, dst_height),
                #     interpolation=cv2.INTER_CUBIC)

                im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # (thresh, im_bw) = cv2.threshold(im_bw, tresh_min, tresh_max, 0)
                # img, contours, hierarchy = cv2.findContours(
                #     im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

                # contour_path = os.path.join(
                #     root_p, file.split(".")[0] + "contours.png")
                # cv2.imwrite(contour_path, image)

                approxes = []

                # high_thresh, thresh_im = cv2.threshold(
                #    im_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # lowThresh = 0.5 * high_thresh

                # edges = cv2.Canny(
                #     im_bw, lowThresh, high_thresh,
                #     apertureSize=3, L2gradient=False)

                low_tres = 100
                edges = cv2.Canny(
                    im_bw, low_tres, low_tres * 3, apertureSize=3)
                lines = cv2.HoughLines(
                    edges, 1, numpy.pi / 180, 400, 0, 0)
                black_image_lines = numpy.zeros_like(image)
                black_image_corners = numpy.zeros_like(image)
                all_lines = []
                for ind in range(lines.shape[0]):
                    for rho, theta in lines[ind]:
                        a = numpy.cos(theta)
                        b = numpy.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + width * (-b))
                        y1 = int(y0 + width * (a))
                        x2 = int(x0 - width * (-b))
                        y2 = int(y0 - width * (a))
                        all_lines.append((x1, y1, x2, y2))

                        cv2.line(
                            black_image_lines,
                            (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.line(
                            image,
                            (x1, y1), (x2, y2), (0, 255, 0), 2)

                edges_path = os.path.join(
                    root_p, file.split(".")[0] +
                    "_edges.png")

                cv2.imwrite(edges_path, edges)

                dst = cv2.cornerHarris(im_bw, 2, 3, 0.04)
                dst = cv2.dilate(dst, None)

                threshold = 0.005 * dst.max()

                black_image_corners[dst > threshold] = [0, 0, 255]

                black_image_all = black_image_lines + black_image_corners

                all_path = os.path.join(
                    root_p, file.split(".")[0] +
                    "_all_good_image.png")

                lines_path = os.path.join(
                    root_p, file.split(".")[0] +
                    "_all_image.png")

                import pdb
                pdb.set_trace()
                # image[numpy.where(
                #     (black_image_all == [0, 255, 255]).all(
                #         axis=2))] = [0, 255, 255]

                # image[black_image_all == [0, 255, 255]] = [0, 255, 255]
                """
                good_lines = []
                for line in all_lines:
                    x1, y1, x2, y2 = line
                    # assert x1 != x2
                    if x1 == x2:
                        points = [(int(x1), int(y)) for y in range(
                            min(y1, y2, 0), max(y1, y2, height))]
                        print(x1)
                    elif y1 == y2:
                        points = [(int(x), y1) for x in range(
                            min(x1, x2, 0), max(x1, x2, width))]
                    else:
                        alpha = (y1 - y2) / (x1 - x2)
                        beta = y1 - ((y1 - y2) * x1 / (x1 - x2))
                        func_line = lambda x: alpha * x + beta
                        points = [(int(x), int(func_line(x))) for x in range(
                            min(x1, x2, 0), max(x1, x2, width))]
                    for point in points:
                        x_p, y_p = point
                        try:
                            pixel = black_image_all[y_p, x_p]
                            if list(pixel) == [0, 255, 255]:
                                good_lines.append(line)
                                cv2.line(
                                    image,
                                    (x1, y1), (x2, y2), (0, 255, 0), 2)
                                break
                        except:
                            pass
                """
                image[numpy.where(
                    (black_image_all == [0, 255, 255]).all(
                        axis=2))] = [0, 255, 255]
                cv2.imwrite(lines_path, image)
                """
                for ind, contour in enumerate(contours):
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    approxes.append(approx)
                    # if len(approx) == 4:
                    #     approxes.append(approx)
                    # if len(contour) == 4:
                    #    approxes.append(contour)

                contour_path = os.path.join(
                    root_p, file.split(".")[0] +
                    "_app_contours.png")
                cv2.drawContours(image, approxes, -1, (0, 255, 0), 3)
                cv2.imwrite(contour_path, image)
                """


if __name__ == "__main__":
    # input example: /data/veles/VD/new_korean/brand_recognition/samples
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", help="Path to the input directory with images to sort.")
    args = parser.parse_args()
    path_to_images = args.input
    get_bboxes_of_tv_brand(path_to_images, 1920, 1080, 128, 255)
