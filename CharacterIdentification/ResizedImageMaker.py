import cv2 as cv
import numpy as np
from CharacterIdentification import ConfigReader
from CharacterIdentification import ImageReader
'''获取载入图片的长宽上限，然后把所有图片等比例resize到其中一边能够达到长宽上限的大小，最后存入regenerate目录'''
max_height = 0
max_width = 0
images = ImageReader.load_images()
print("image load in complete")
for chars in images:
    for typeface in chars:
        h, w = typeface.shape
        if h > max_height:
            max_height = h
        if w > max_width:
            max_width = w
regenerated_images = []
char_idx = 0
wirtten_root = "F:\\dataset\\chiese_characters\\cnftl-20171119\\regenerated\\"
for chars in images:
    row = []
    typeface_idx = 0
    for typeface in chars:
        # 使用vstack和hstack为image按照max h 和 w的大小扩展边界
        h, w = typeface.shape
        if (max_height - h) % 2 != 0:
            upper_side_h = (max_height - h) // 2
            lower_side_h = upper_side_h + 1
        else:
            upper_side_h = (max_height-h) // 2
            lower_side_h = upper_side_h
        typeface = np.vstack((typeface, np.zeros((upper_side_h, w))))
        typeface = np.vstack((np.zeros((lower_side_h,w)),typeface))
        if (max_width - w) % 2 != 0:
            right_side_w = (max_width - w) // 2
            left_side_w = right_side_w + 1
        else:
            right_side_w = (max_width - w) // 2
            left_side_w = right_side_w
        typeface = np.hstack((np.zeros((max_height, right_side_w)), typeface))
        typeface = np.hstack((typeface, np.zeros((max_height, left_side_w))))
        cv.imwrite(wirtten_root+str(char_idx)+"\\"+str(typeface_idx)+".png", typeface)
        row.append(typeface)
        typeface_idx += 1
    regenerated_images.append(row)
    char_idx += 1
print("image written complete")
pass