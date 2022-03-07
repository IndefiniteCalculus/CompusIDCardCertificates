import cv2 as cv
import numpy as np
import os
from CharacterIdentification import ConfigReader as conf
from CharacterIdentification import ImageReader,LabelReader
def load_images(load_in_range:tuple = None):
    images = []
    # find root dir of images
    dir, _ = conf.get_dir_Chinese_Characters()
    dir = dir + "\\regenerated"
    char_types = [str(i) for i in range(3982)]
    if load_in_range is None:
        load_in_range = (0, float('inf'))
    idx = 0
    for char_type in char_types:
        dir_char = dir + "\\" + char_type
        typefaces = os.listdir(dir_char)
        char = []
        if idx >= load_in_range[0] and idx <= load_in_range[1]:
            pass
            for typeface in typefaces:
                # if typeface != 'STCAIYUN.TTF.png' \
                #         and typeface != 'STLITI.TTF.png' \
                #         and typeface != 'STXINGKA.TTF.png' \
                #         and typeface != 'STHUPO.TTF.png':
                if typeface == '4.png':
                    dir_typeface = dir_char + "\\" + typeface
                    # cv.imshow(dir_typeface, cv.imread(dir_typeface))
                    # cv.waitKey(500)
                    im = cv.imread(dir_typeface,0) # read in gray scale image
                    char.append(im)

            if idx%300 == 0:
                print(str(idx)+" image loaded")
            images.append(char)
        idx += 1
    return images
label = LabelReader.get_labels()
im = load_images((0,float('inf')))
idx = 0
written_dir = r"F:\dataset\chiese_characters\cnftl-20171119\tiff"
for row in im:
    for char in row:
        if idx >=15 and idx <=24 or idx >=108:
            file_name = written_dir + "\\" + label[idx] + ".png"
            cv.cv2.imencode('.tiff', char)[1].tofile(file_name)
    idx += 1
pass