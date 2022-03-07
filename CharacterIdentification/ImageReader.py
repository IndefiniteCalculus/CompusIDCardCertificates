import os
import cv2 as cv
import numpy as np
from CharacterIdentification import ConfigReader as conf
def load_images(load_in_range:tuple = None):
    images = []
    # find root dir of images
    dir, _ = conf.get_dir_Chinese_Characters()
    dir = dir + "\\train"
    char_types = os.listdir(dir)
    if load_in_range is None:
        load_in_range = (0, float('inf'))
    idx = 0
    for char_type in char_types:
        dir_char = dir + "\\" + char_type
        typefaces = os.listdir(dir_char)
        char = []
        if idx >= load_in_range[0] and idx <= load_in_range[1]:
            for typeface in typefaces:
                if typeface != 'STCAIYUN.TTF.png' \
                        and typeface != 'STLITI.TTF.png' \
                        and typeface != 'STXINGKA.TTF.png' \
                        and typeface != 'STHUPO.TTF.png':
                    dir_typeface = dir_char + "\\" + typeface
                    # cv.imshow(dir_typeface, cv.imread(dir_typeface))
                    # cv.waitKey(500)
                    im = cv.imread(dir_typeface,0) # read in gray scale image
                    char.append(255 - im)
            idx += 1
            if idx%300 == 0:
                print(str(idx)+" image loaded")
            images.append(char)
    return images
if __name__ == '__main__':
    print("ImageReader test begin")
    images = load_images((0,1000))
    pass