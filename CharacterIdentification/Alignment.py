import cv2 as cv
import numpy as np

def Alignment(image:np.ndarray):
    src_image = image.copy()
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    black_image = np.zeros(image.shape[:2])
    # 将图像调整至水平
    # 平滑处理图像，设置自适应阈值算法
    image = cv.GaussianBlur(image,(3,3),1)
    edge = cv.Canny(image, 100, 255)

    # cv.imshow('edge', edge)
    # cv.waitKey(0)
    minArea_rectangle = border_angle(edge)
    image = mask_rotate_crop_resize(minArea_rectangle, src_image)
    # cv.imshow("first aligned image",image)
    # cv.waitKey(0)
    # 旋转后的图像，可能仍有边缘，进一步简单处理一下
    # edge = cv.Canny(image,100,255)
    # cv.imshow("aligned edge",edge)
    # cv.waitKey(0)
    # _, contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # box = cv.boundingRect(edge)
    # image_without_borders = src_image[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]
    return image

def mask_rotate_crop_resize(rect:tuple, src_image):
    '''use mask get rid of noice outside of borders,
    rotate image to horizontal, ###and crop it, then resize it to fixed size( h / w = 0.6 )'''
    # cv.imshow("image",src_image)
    center, theta = rect[0], (rect[2] + 90) % 90
    (h, w) = src_image.shape[:2]
    mask = np.zeros(src_image.shape[0:2])
    mask = mask.astype("uint8")
    src_image = src_image.astype("uint8")
    # draw borders of src_image
    box_points = cv.boxPoints(rect)
    cv.line(mask, tuple(box_points[0]),tuple(box_points[1]),color=(255))
    cv.line(mask, tuple(box_points[1]), tuple(box_points[2]), color=(255))
    cv.line(mask, tuple(box_points[2]), tuple(box_points[3]), color=(255))
    cv.line(mask, tuple(box_points[3]), tuple(box_points[0]), color=(255))
    borders, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.fillConvexPoly(mask,borders[0],(1))
    # cv.imshow("mask", mask)
    # cv.waitKey(0)
    for channel in range(src_image.shape[-1]):
        src_image[:,:,channel] = src_image[:,:,channel] * mask
    # cv.imshow("masked src", src_image)
    M = cv.getRotationMatrix2D(center, theta, 1)
    rotated_image = cv.warpAffine(src_image, M, (int(w), int(h)), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
    # cv.imshow("rotated image",rotated_image)
    box = cv.boundingRect(cv.cvtColor(rotated_image, cv.COLOR_RGB2GRAY))
    image_without_borders = rotated_image[box[1] +5:box[1]+box[3] -5 , box[0] + 5 :box[0]+box[2] - 5, : ]
    # cv.imshow("image with out borders", image_without_borders)
    # cv.waitKey(0)
    return image_without_borders


def rotate(image, theta, center):
    (h, w) = image.shape[:2]
    M = cv.getRotationMatrix2D(center, theta, 1)
    rotated_image = cv.warpAffine(image, M, (int(w), int (h)), cv.INTER_LINEAR, borderMode= cv.BORDER_CONSTANT, borderValue=(255))
    return rotated_image

def show_edge(contours, backgroud):
    '''依次延迟显示边缘集合中的各个边缘'''
    for idx in range(len(contours)):
        cv.drawContours(backgroud, contours, idx, (255))
        cv.imshow("a", backgroud)
        cv.waitKey(1000)

def border_angle(edge:np.ndarray) -> ((),(),()):
    '''get area image should be croped and rotate angle'''
    edge = edge.astype("uint8")
    demostrate_image = np.zeros(edge.shape[:2])
    # 用形态学操作膨胀，把各个边连接为几个，用findContunus找到边缘图中各个边缘的父子关系
    contours, hierarchy = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    n_contours = len(contours)
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3,3))
    # cv.imshow("pre-process edge", edge)
    while n_contours > 8:
        edge = cv.dilate(edge, kernel, iterations = 1)
        contours, hierarchy = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        n_contours = len(contours)
        pass
    # cv.imshow("processed edge", edge)
    # cv.waitKey(0)
    # 依次找各个边缘, 方法是判断每个边缘的最小外接矩形面积是否超过了图像总面积的0.3，是的话就把边缘加入到边缘点集合中
    borders = None
    image_area = edge.shape[0] * edge.shape[1]
    for contour in contours:
        rect = cv.minAreaRect(contour)
        h = rect[1][0]
        w = rect[1][1]
        area = h * w
        if h * w > 0.2 * image_area:
            if borders is None:
                borders = contour
            else:
                borders = np.array(borders.tolist() + contour.tolist())
                # np.hstack((borders, contour))
            # show_edge([borders],demostrate_image)
        pass
    cv.drawContours(demostrate_image, borders, -1,255)
    # cv.imshow("founded contours", demostrate_image)
    # cv.waitKey(0)
    # show_edge([borders],backgroud=demostrate_image)
    # 用minAreaRect得到边缘矩阵的最小旋转角度
    rect = cv.minAreaRect(borders)
    # box = cv.boxPoints(rect)
    # cv.line(demostrate_image, tuple(box[0]),tuple(box[1]),color=(255))
    # cv.line(demostrate_image, tuple(box[1]), tuple(box[2]), color=(255))
    # cv.line(demostrate_image, tuple(box[2]), tuple(box[3]), color=(255))
    # cv.line(demostrate_image, tuple(box[3]), tuple(box[0]), color=(255))
    # cv.imshow("rect found",demostrate_image)
    # demostrate_image = rotate(demostrate_image, (90+rect[-1])%90, center=rect[0])
    # cv.imshow('rect rotated', demostrate_image)
    # cv.waitKey(0)
    return rect # (
    # (center point x, center point y),
    # (width, height),
    # degree value can turn rectangle to horizontal in clockwise
    # )
