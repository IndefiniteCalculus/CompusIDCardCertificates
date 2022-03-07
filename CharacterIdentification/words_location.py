'''用canny算子扫出边沿后再在原图像上做投影把行切出来'''
import cv2 as cv
import numpy as np

import CharacterIdentification.ConfigReader as conf
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
from CharacterIdentification import feature_type
# def feature_extract(src_image):
#     what_feature = feature_type.what_feature()
#     image = cv.cvtColor(src_image, cv.COLOR_RGB2GRAY)
#     return extract_char_images(image, what_feature)
#     pass

def hist_test(edge):
    x = np.sum(edge, axis=1)
    background = np.zeros(edge.shape)
    for i in range(edge.shape[0]):
        cv.line(background, (0,i), (int(x[i]), i), (255))
    cv.imshow("hist",background)
    cv.waitKey(0)
    pass

def extract_char_images(src_image:np.array,split = True):
    '''project each row of card to find out where line has characters is'''
    # use edge area to find out where words are(not ideal enough, may influenced by background)
    # edge = cv.Canny(image, 100, 255)
    # kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3))
    # edge = cv.morphologyEx(edge, cv.MORPH_DILATE, kernel, iterations=1)
    # _, contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # show_edge(contours, edge.shape)
    # cv.imshow("dilate", edge)
    #
    # use color to identified words, and use face recognition to wipe out influence on projection by  image of face
    image = cv.cvtColor(src_image,cv.COLOR_RGB2GRAY)
    # src_image = cv.equalizeHist(src_image)
    background = np.ndarray(image.shape)
    background[:,:] = 0

    idx = image < 48 # !!!这个参数对与切分字符的操作非常敏感，建议往大了调，避免把数字切开了合起来麻烦，尝试修改为自适应的参数
    background[idx] = 255#image[idx]

    # find right side
    # 1. use face recognition, but may still remain some noise from other part face photo
    # dir, names = conf.get_dir_opencvmodel()
    # classifier = cv.CascadeClassifier(dir + "//" + names[0])
    # faces = classifier.detectMultiScale(image)
    # if len(faces) == 1:
    #     x, y, w, h = faces[0]
    #     right_side = x + w
    #     words_area = background[:,right_side:]
    # else:
    #     exit("no avaliable face detected")
    #
    # 2. use sobel find the longest edge alone x axis of photo box，
    # view a contour that it's bounding box high enough
    # and horizontal clustering center closed to the middle of image,
    # treat it's bounding box clustering center as the separation of photo and words
    image = cv.Canny(image, 100, 255)
    sobel = cv.Sobel(image, cv.CV_8U, dx = 1, dy = 0)
    sobel = cv.Sobel(sobel, cv.CV_8U, dx=1, dy=0)
    sobel = cv.Sobel(sobel, cv.CV_8U, dx=1, dy=0)
    # cv.imshow('sobel', image)
    # cv.waitKey(0)
    contours, _ = cv.findContours(sobel, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    center_x = image.shape[1] / 2
    min_dist = center_x
    target_x = None
    for contour in contours:
        x, y ,w, h = cv.boundingRect(contour)
        if h / image.shape[0] > 0.3 :
            if min_dist > abs(center_x - (x + w/2)):
                target_x = int(x + w/2)
                min_dist = abs(center_x - (x + w/2))
    if target_x is None:
        target_x = int(center_x)
    words_area = background[:, target_x:]
    words_src_area = src_image[:,target_x:]
    # cv.imshow('word_src_area', words_src_area)
    # cv.waitKey(0)
    if split == False:
        return words_area
    # # we got the words area, so another side should be the face area,
    # # find a contours which has largest area,
    # # the height of it should be considered as height range of words area

    face_area = image[:,:target_x]
    face_area = cv.Canny(face_area, 100, 255)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(3,3))
    face_area = cv.morphologyEx(face_area, cv.MORPH_DILATE, kernel,iterations=0)
    contours, _ = cv.findContours(face_area, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    area = 0
    y_range = None
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        if w*h > area:
            largest_contour = contour
            y_range = (y, y + h)
            area = w*h
    if y_range is None:
        exit("no available card detected")
    # black = np.zeros(face_area.shape)
    # cv.drawContours(black, [largest_contour], -1, (255))
    # cv.imshow("larget contour",black)
    # cv.waitKey(0)

    # now use dilate operation to enhance thickness of words, make projection of image be identified easier
    words_area = words_area[y_range[0]:y_range[1], :]
    words_src_area = words_src_area[y_range[0]:y_range[1], :]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    thicker_words = cv.morphologyEx(words_area, cv.MORPH_DILATE, kernel)
    # project image alone the axis 1, and get the range of each row
    porjection  = np.sum(thicker_words, axis=1)
    idx = porjection > 0
    up_side, down_side = 0,0
    range_list = []
    height = y_range[1] - y_range[0]
    for k in range(len(idx)-1):
        if idx[k] == False and idx[k+1] == True:
            up_side = k + 1
        if idx[k] == True and idx[k+1] == False:
            down_side = k + 1
            h = down_side - up_side
            rate = h / height
            if rate > 0.05 and rate < 0.2:
                range_list.append((up_side, down_side))
    image_list = []
    r_id = 0
    noise_remove_edge = int(words_area.shape[1] / 50)
    for h_range in range_list:
        splited_mask_image = words_area[h_range[0]:h_range[1],noise_remove_edge:]
        splited_src_image = words_src_area[h_range[0]:h_range[1],noise_remove_edge:]
        # erode some noise and dilate back(may not necessary)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        # noice_removed_row = cv.morphologyEx(splited_image, cv.MORPH_OPEN, kernel)
        char_projection = np.sum(splited_mask_image, axis=0)
        up_side, down_side = 0,0
        each_chars = []
        c_id = 0
        for pixel in range(splited_mask_image.shape[1]-1):
            if char_projection[pixel] == 0 and char_projection[pixel + 1] != 0:
                up_side = pixel + 1
            if char_projection[pixel] != 0 and char_projection[pixel + 1] == 0:
                down_side = pixel + 1 # numpy索引时是左闭右开区间
                each_chars.append(splited_src_image[:,up_side:down_side])
                # cv.imshow(str((r_id, c_id)), splited_image[:,up_side:down_side])
                c_id += 1
                # cv.waitKey(1000)
        r_id += 1
        image_list.append(each_chars)
        # cv.imshow(str(h_range), splited_image)

    # 进一步处理字符图像，任务有三
    # 1、将可能的噪声信号去除（这步其实应该在前面切片的时候完成）考虑设为0.02
    # 2、将可能把偏旁切出来了的字符合并成原来的样子，选择该行字符长度中位数为标杆字符
    # （通过比较字符长度是否为标杆字符长度的50%判断是否为偏旁或冒号），注意判断是左偏旁还是右偏旁（左右合并后，判断长度与标杆的误差是否在22%以内）并注意区分冒号
    # 3、尝试把第四行的数字后面或者前面跟的横线去掉，一般直接对半切
    # 为避免数字长度不一造成的影响，只对第二行和第三行使用以上方法合并字符
    # 不要想着去判断大的那一边的偏旁，有些本体就和某些字的偏旁一样大
    # 噪声会影响对较小偏旁和冒号的判断，请务必确保已经去除全部噪声（考虑使用固定值）
    # 字符和噪声的区分：标杆字符长度的误差在xxx以内为完整字符（在此之前切片的时候进行处理，考虑对其投影后的波峰进行处理）
    # 偏旁的判断：标杆字符长度的0.5到0.15
    # 冒号的判断：偏旁判断成立的基础上加入大于0像素比例xxx（待实验）偏旁和冒号在像素率上的区分不明显，考虑使用行列直接排除掉冒号
    chars_list = []
    row_id = 0
    for each_row in image_list:
        width_list = []
        for each_char in each_row:
            width_list.append(each_char.shape[1])
        standerd = np.median(width_list)
        chars_row = []
        col_id = 0
        row_len = len(each_row)
        merged_char_idx = [] # (idx, -1)means merged with someone before the idx, (idx, 1)means merged with someone after the idx, used by pop(value = each_row[idx+-1)
        for each_char in each_row:
            if row_id == 0:
                # 这里最好再加一个噪声去除方法
                chars_row.append(each_char)
                col_id += 1

            if row_id == 1 or row_id == 2:
                char_error = abs(standerd - each_char.shape[1])
                if char_error / standerd < 0.2:
                    chars_row.append(each_char)
                else:
                    if char_error / standerd >= 0.2 and (
                            (row_id == 1 and col_id != 2) or (row_id == 2 and col_id != 4)
                    ):
                        # maybe it's a splited side from some char, try merge it with someone near by
                        # ps: remenber to kick the one out of chars_row if successfully merged with someone
                        merged_char = None
                        merged_with = None
                        if row_id == 1 or row_id == 2:
                            if col_id == 3 and row_id == 1 or col_id == 5 and row_id == 2:
                                # 只需和右边的合并
                                merged_char = np.hstack((each_char, each_row[col_id+1]))
                                merged_with = 1
                            elif col_id == row_len -1:
                                merged_char = np.hstack((each_row[col_id -1], each_char))
                                merged_with = - 1
                            else:
                                merged_char0, merged_char1 = np.hstack((each_row[col_id-1],each_char)), np.hstack((each_char,each_row[col_id+1]))
                                char_error0, char_error1 = abs(merged_char0.shape[1] - standerd), abs(merged_char1.shape[1] - standerd)
                                if char_error0 < char_error1:
                                    merged_char = merged_char0
                                    merged_with = - 1
                                else:
                                    merged_char = merged_char1
                                    merged_with = 1

                        if abs(merged_char.shape[1] - standerd) / standerd < 0.7:
                            chars_row.append(merged_char)
                            merged_char_idx.append((col_id, merged_with))
                        else:
                            chars_row.append(each_char)
                        pass
                    else:
                        chars_row.append(each_char)
                col_id += 1

            if row_id == 3:
                # 对可能带尾巴的字符进行切分，注意排除标题部分
                char_error = abs(standerd - each_char.shape[1])
                if char_error / standerd > 0.7 and col_id > 4:
                    w = each_char.shape[1]
                    split_char0, split_char1 = each_char[:,:w // 2], each_char[:,w //2:]
                    chars_row.append(split_char0)
                    chars_row.append(split_char1)
                else:
                    chars_row.append(each_char)
                col_id += 1
        row_id += 1
        deleted = 0
        if len(merged_char_idx) != 0:
            # 有字符合并出现
            i = 0
            while i < len(merged_char_idx):
                if i != len(merged_char_idx) - 1:
                    # 对于每一个发生过合并的字符，如果有重复合并发生，将后一个合并字符与其合并记录删去
                    if merged_char_idx[i][1] == -merged_char_idx[i+1][1]:
                        merged_char_idx.pop(i+1)
                        chars_row.pop(merged_char_idx[i][0] - deleted)
                        deleted += 1
                        i -= 1
                i += 1
        chars_list.append(chars_row)
    return  chars_list
    # show_image_list(chars_list)
    # extract features from images
    # features = []
    # for each_row in chars_list:
    #     p_row = []
    #     for each_char in each_row:
    #         # normalization
    #         max_v = np.max(each_char)
    #         min_v = np.min(each_char)
    #         N_im = (each_char - min_v) / (max_v - min_v)
    #         # 选择提取什么样的特征
    #         if what_feature == 'projection':
    #             p_row.append(extract_pojections(N_im))
    #         elif what_feature == 'knn':
    #             p_row.append(extract_knn(N_im))
    #         elif what_feature == 'pca':
    #             p_row.append(extract_pca(N_im))
    #         else:
    #             exit("unavailable feature")
    #     features+=p_row
    # return features, chars_list

# 归一化方法，仅投影法
def normalization(image:np.array):
    # 先将图片等比例缩放并分别将每个边调整到20
    h_w_rate = image.shape[0] / image.shape[1]
    h_rate = 20 / image.shape[0]
    w_rate = 10 / image.shape[1]
    h_image = cv.resize(image, dsize=(0, 0), fx=h_rate, fy=h_rate)
    w_image = cv.resize(image, dsize=(0, 0), fx=w_rate, fy=w_rate)
    return h_image,w_image,h_rate, w_rate

def extract_pojections(image:np.array):
    '''extract horizontal and vertical projections
    -->[projection0, projection1]'''
    # 先将图片等比例缩放并分别将每个边调整到20
    h_image, w_image,h_rate,w_rate = normalization(image)
    projection1 = np.sum(h_image, axis=1)
    projection0 = np.sum(w_image, axis=0)
    return [projection0, projection1]

def extract_knn(image:np.array):
    '''flatten an image to a 1-D vector'''
    image = cv.resize(image,(20,20))
    return image.reshape(1,-1)

def extract_pca(image:np.array):
    image = cv.resize(image, (20,20))
    vector_im = image.reshape(-1,1)
    return vector_im

def show_image_list(image_list):
    row = 0
    col = 0
    for each_row in image_list:
        for each_char in each_row:
            cv.imshow(str((row,col)),each_char)
            cv.waitKey(800)
            col += 1
        row += 1
        col = 0
    cv.waitKey(0)


def show_edge(contours, shape):
    '''依次延迟显示边缘集合中的各个边缘'''
    for idx in range(len(contours)):
        x,y,w,h = cv.boundingRect(contours[idx])
        area = w * h
        backgroud = np.zeros(shape)
        if area > shape[0] * shape[1] / 5 and area < shape[0] * shape[1] / 4:
            cv.drawContours(backgroud, contours, idx, (255))
            cv.imshow(str(area/shape[0] /shape[1]), backgroud)
            cv.waitKey(300)
            # cv.destroyWindow(str(area))
