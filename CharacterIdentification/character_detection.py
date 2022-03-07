from CharacterIdentification import ConfigReader as cfg
import cv2 as cv
from CharacterIdentification import Alignment as alg
from CharacterIdentification import words_location
from SystemConponents import SQLTool
import pytesseract
import re
import numpy as np
def detect(image_dir:str = None):
    '''返回识别到的字符串列表，如果没在数据库中或者没有有效字段就返回False'''
    if image_dir is None:
        '''对图像进行预处理，目标是把图像调整至水平，缩放处理后分割出图像上的文字'''
        dir , images_name = cfg.get_dir_test_image()
        # 将图像灰度化，并使用锐化增强图像显示
        image_dir = dir + "\\" + "certificate_rotated0.jpg"
        image_dir = image_dir.encode('gbk').decode()
    image = cv.imread(image_dir)
    # 注意这里最好加一个自适应的resize方法，把图像缩放到合适的大小，确保图像中存在足够多的像素点
    image = cv.resize(image,dsize=(0,0), fx = 1/3, fy = 1/3)
    # cv.imshow("src image", image)
    # 将图像调整至水平(已实现), 确定出卡的位置（已实现）
    aligned_image = alg.Alignment(image)
    # cv.imshow("aligned image", aligned_image)
    # cv.waitKey(0)
    # 切割出文字（已实现）
    # chars_list = words_location.extract_char_images(aligned_image)
    # idx = 0
    # text_context = []
    # for row in chars_list:
    #     for char in row:
    #         # char = cv.resize(char,(100,100))
    #         char = cv.resize(char,(0,0), fx = 2, fy = 2)
    #         char = char.astype('uint8')
    #         char = cv.cvtColor(char, cv.COLOR_RGB2GRAY)
    #         # char = cv.equalizeHist(char)
    #         _,char = cv.threshold(char,0,255,cv.THRESH_OTSU)
    #         # char = cv.morphologyEx(char,cv.MORPH_CLOSE, kernel, iterations=1)
    #         # char[char > 0] = 255
    #         cv.imwrite('F:\\dataset\\campus_card\\splited\\' + str(idx) + '.png', char)
    #         # cv.imshow(str(idx), char)
    #         idx += 1
    #         # cv.waitKey(100)
    #         pass
    # cv.waitKey(0)
    # pass
    '''特征工程部分，目标是载入训练图片和其标签，尽可能无损地将大小归一化处理（图像大小）
    并确定一种分类方法将待查图片和训练图对应起来，将标签作为识别结果返回'''
    # 特征提取和特征比较分类是深度耦合的，不要分开，按特征提取比较的方法来分
    # images = ImageReader.load_images((0,1000))
    # labels = LabelReader.get_labels()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # 归一化处理，原则是以chars_list的最大长宽确定等比例缩放的最终大小（等比例缩放）
    area = words_location.extract_char_images(aligned_image, split=False)
    tessdata_dir_config = '--tessdata-dir "F:\\models\\tessdata\\tessdata_best-master"'
    area = cv.morphologyEx(area, cv.MORPH_CLOSE,kernel)
    # cv.imshow("words_area", area)
    # cv.waitKey(0)
    text = pytesseract.image_to_string(area.astype('uint8'),lang='chi_sim',config=tessdata_dir_config)
    # print(text)
    text = text.replace(' ', '')
    text = text.split('\n')
    # 开始正则匹配
    student_id_matcher = re.compile(r'20(\d){8}')
    name_matcher = re.compile(SQLTool.get_distinct_name())
    college_matcher = re.compile(SQLTool.get_distinct_college())
    uid_matcher = re.compile(SQLTool.get_distinct_uid())
    matchers = [student_id_matcher, name_matcher, college_matcher, uid_matcher]
    last_matched_row = 0
    results = []
    for matcher in matchers:
        # row_id = 0
        result_each_matcher = []
        for row in text:
            # if last_matched_row < row_id:
                result = matcher.search(row)
                if result is not None :
                    result = result.span()
                    result = row[result[0]: result[1]]
                    if len(result) != 0:
                        # last_matched_row = row_id
                        result_each_matcher.append(result)
            # row_id += 1
        results.append(result_each_matcher)
        pass

    info = None
    # 验证方法可克服未识别到的情况进入下一步的识别
    # 验证学号
    std_stus = SQLTool.get_distinct_stuid(False)
    max_stu_rate, max_pos , stu_rates = evaluate(std_stus,results[0])
    info = SQLTool.get_one_person(std_stus[max_pos[0]])
    if max_stu_rate < 0.7 and max_stu_rate > 0.5:
        # 学号了解了，进一步验证姓名
        std_name = info[1]
        names = results[1]
        max_name_rate, max_name_pos, name_rates = evaluate([std_name], names)
        if max_name_rate < 0.6 :
            #姓名不明，进一步验证学院名
            std_college = info[2]
            colleges = results[2]
            max_college_rate, max_college_pos, college_rates = evaluate([std_college], colleges)
            if max_college_rate < 0.8:
                # no available card in the database
                return False
    elif max_stu_rate <=0.5:
        # 识别不到学号找不到人，判定为无效
        return False
    return info, aligned_image

def evaluate(gallary_set:list, quary_set:list, find_continuity:bool = True):
    '''gallary ((),(),(),) quary []
    -->max_rate
    __>max_rate position in rate of list, (gallary_id, quary_id)
    -->rate of list [ [quary1-gallary1, quary2-gallary1,], [quary1-gallary2, quary2-gallary2, ], ]'''
    rates = []
    max_rate = 0
    gallary_idx, quary_idx = 0,0
    max_position = (0,0)
    for gallary in gallary_set:
        quary_rate = []
        for quary in quary_set:
            if find_continuity == True:
                common_stuid_len = find_common_substr(quary, gallary)
            else:
                common_stuid_len = find_common_char(quary, gallary)
            rate = common_stuid_len / len(gallary)
            if rate > max_rate:
                max_rate = rate
                max_position = (gallary_idx, quary_idx)
            quary_rate.append(rate)
            quary_idx += 1
        rates.append(quary_rate)
        gallary_idx += 1
    return max_rate,max_position, rates

def find_common_char(str1, str2):
    str1, str2 = set(str1), set(str2)
    return len(str1 & str2)

def find_common_substr(str1, str2):
    count = 0
    length = len(str1)
    for sublen in range(length, 0, -1):
        for start in range(0, length - sublen + 1):
            count += 1
            substr = str1[start:start + sublen]
            if str2.find(substr) > -1:
                return len(substr)
    else:
        return 0
# chars_list = words_location.extract_char_images(aligned_image)
# idx = 0
# text_context = []
# for row in chars_list:
#     for char in row:
#         # char = cv.resize(char,(100,100))
#         char = cv.resize(char,(0,0), fx = 2, fy = 2)
#         char = char.astype('uint8')
#         char = cv.cvtColor(char, cv.COLOR_RGB2GRAY)
#         # char = cv.equalizeHist(char)
#         _,char = cv.threshold(char,0,255,cv.THRESH_OTSU)
#         # char = cv.morphologyEx(char,cv.MORPH_CLOSE, kernel, iterations=1)
#         # char[char > 0] = 255
#         text = pytesseract.image_to_string(char,lang='chi_sim',
#                                            config=tessdata_dir_config +
#                                           "--c tessedit_char_whitelist=0123456789江笑语学号所属学院光电工程学院 --psm 6")
#         text_context.append(text)
#         cv.imshow(str(idx), char)
#         idx += 1
#         cv.waitKey(100)
#         pass
# pass

# model = ChineseCharNet.ChineseCharNet()
# model.load_state_dict(torch.load("chinese_char_model_100.pt"))
# idx = 0
# for row in chars_list:
#     for char in row:
#         char = cv.resize(char,(10,10))
#         input_char = np.reshape(char,(1,1,10,10))
#         y_pred = model.forward(torch.tensor(input_char).float())
#         y_pred = y_pred.data.numpy()
#         print(LabelReader.decode_label(y_pred))
#         cv.imshow(str(idx),char)
#         print(idx)
#         cv.waitKey(1000)
# FeatureComparison.test_dist(quary_proj, gallary_proj, labels)
# cv.waitKey(0)
if __name__ == '__main__':
    text = detect()
    pass