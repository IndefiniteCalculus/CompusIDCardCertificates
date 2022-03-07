from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
import tkinter.filedialog as filedialog
from SystemConponents import ObtainFaceImage, SQLTool
from CharacterIdentification import character_detection
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
from FaceIdentification.model import compare
tf.disable_v2_behavior()
# dist = compare.cmp(img1, img2)
# print(dist)

# to make sure all the variable keep existing, make a class
class window():
    def __init__(self):
        '''initialization GUI component in windows'''
        self.root = Tk()
        self.root.title("一卡通识别系统")
        self.w_max, self.h_max = self.root.maxsize()
        self.root.geometry("%sx%s" % self.get_windows_w_h())
        self.camera = cv.VideoCapture(0)
        self.place_first_level_component()
        self.bool_keep_photo = False # keep taking photos
        self.bool_photo_ready1 = False # if card photo ready
        self.bool_photo_ready2 = False # if face photo ready
        self.bool_photo_compared = False # used for text show formate
        self.root.mainloop()

    def get_windows_w_h(self):
        self.w_window, self.h_window = int(self.w_max * 3 // 4), int(self.h_max * 2 // 3)
        return (self.w_window, self.h_window)

    def get_1_component_w_h(self):
        self.w_comp_1, self.h_comp_1 = int(self.w_window * 1 // 4), int(self.h_window * 3 // 4)
        return (self.w_comp_1, self.h_comp_1)

    def place_first_level_component(self):
        # initial first level component
        w, h = self.get_1_component_w_h()
        self.canvas_card_image, self.canvas_face_image, self.info_place, self.pic_buttom = \
        Canvas(width = w, height = h, bg = 'white'), \
        Canvas(width = w, height = h, bg = 'white'),\
        Text(width = w * 4 // 32, height = h * 2 // 32, bg = 'white'),\
        Button(self.root, width = 12, height = 5, bg = 'red', command = self.start_camera)
        # place componets
        anchor_points_w = [self.w_window // 16, self.w_window * 3 // 8, self.w_window * 11 / 16]
        anchor_points_h = [self.h_window // 16, self.h_window // 16, self.h_window // 16]
        self.canvas_card_image.place(x = anchor_points_w[0], y = anchor_points_h[0])
        self.canvas_face_image.place(x = anchor_points_w[1], y = anchor_points_h[1])
        self.info_place.place(x = anchor_points_w[2], y = anchor_points_h[2])
        self.pic_buttom.place(x = self.w_window * 15 // 32, y = self.h_window * 27 // 32)
        # place menubar
        self.menubar = Menu(self.root)
        self.menu_fileoption = Menu(self.menubar)
        self.menu_fileoption.add_command(label="导入证件照",command = self.load_card_photo)
        self.menubar.add_cascade(label='文件',menu=self.menu_fileoption)
        self.root.config(menu=self.menubar)
        # # test usage, show anchor points
        # self.component1.create_rectangle(-10, -10, 10, 10, fill="red", outline="black")
        # self.component2.create_rectangle(-10,-10,10,10, fill="red", outline="black")
        # self.component3.create_rectangle(-10,-10,10,10, fill="red", outline="black")
        pass

    def load_card_photo(self):
        '''从文件浏览器导入'''
        self.card_image = self.open_im_dir()
        '''可以开始识别文字了'''
        self.bool_photo_ready1 = True# card image has been loaded
        result = character_detection.detect(self.dir_card)
        if result is  False:
            text, self.aligned_card_image = 'false', np.zeros((0,0))
            self.info_place.delete(1.0, END)
            self.info_place.insert(1.0, text)
        else:
            text, self.aligned_card_image = result
            self.update_card_photo()
            self.info_place.delete(1.0,END)
            title = ["学号：","姓名：","学院：","认证码："]
            for line_idx in range(len(text)):
                self.info_place.insert(END,title[line_idx]+text[line_idx]+"\n")
            self.info_place.delete(END, END)
            if self.bool_photo_ready2 == True:
                self.score_face = compare.cmp(self.aligned_card_image, self.face_image)
                if self.score_face > 0.9:
                    self.conclusion = '不是同一个人'
                else:
                    self.conclusion = '是同一个人'
                if self.bool_photo_compared == True:
                    self.info_place.delete(5.0, END)
                else:
                    self.bool_photo_compared = True
                self.info_place.insert(END, '\n'+str(self.conclusion))
                print(self.score_face)



    def start_camera(self):
        if self.bool_keep_photo == False:
            after = self.root.after(200, self.camera_get_photo)
            self.bool_keep_photo = True
        else:
            self.bool_keep_photo = False # notify thread to stop
            self.bool_photo_ready2 = True# face image has been loaded
            if self.bool_photo_ready1 == True:
                #识别
                self.score_face = compare.cmp(self.aligned_card_image, self.face_image)
                if self.score_face > 0.9:
                    self.conclusion = '不是同一个人'
                else:
                    self.conclusion = '是同一个人'
                if self.bool_photo_compared == True:
                    self.info_place.delete(5.0, END)
                else:
                    self.bool_photo_compared = True
                self.info_place.insert(END, '\n'+str(self.conclusion))
                print(self.score_face)

    def camera_get_photo(self):
        self.load_face_photo()
        self.update_face_photo()
        if self.bool_keep_photo == True:
            self.root.after(200, self.camera_get_photo)
        else:
            pass

    def load_face_photo(self):
        '''从摄像头导入'''
        if self.camera.isOpened():
            self.face_image = ObtainFaceImage.get_image(self.camera)

    def update_face_photo(self):
        # resize to match canvas size
        resize_rate = self.w_comp_1 / self.face_image.shape[1]
        image = cv.resize(self.face_image, (0, 0), fx=resize_rate, fy=resize_rate)
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        # transform image to canvas
        image = Image.fromarray(image)
        self.face_imTk = ImageTk.PhotoImage(image=image)
        self.canvas_face_image.create_image(self.w_comp_1//2, self.h_comp_1 // 2, anchor=CENTER, image=self.face_imTk)
        # self.canvas_face_image.update()

    def update_card_photo(self):
        # resize to match canvas size
        resize_rate = self.w_comp_1 / self.card_image.shape[1]
        image = cv.resize(self.card_image, (0, 0), fx=resize_rate, fy=resize_rate)
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        # transform image to canvas
        image = Image.fromarray(image)
        self.card_imTk = ImageTk.PhotoImage(image=image)
        rate = self.w_comp_1 / self.aligned_card_image.shape[1]
        aligned = cv.resize(self.aligned_card_image,(0,0),fx = rate, fy = rate)
        self.aligned_card_imTk = im_np2im_tk(aligned)
        self.canvas_card_image.create_image(0, 0, anchor=NW, image=self.card_imTk)
        self.canvas_card_image.create_image(0, self.h_comp_1,anchor=SW, image = self.aligned_card_imTk)

    def open_im_dir(self)->np.array:
        files = [("PNG图片", "*.png"), ("JPG(JEPG)图片","*.j[e]{0,1}pg"), ("所有文件", "*")]
        self.dir_card = filedialog.askopenfilename(title="选择图片", filetypes=files)
        if len(self.dir_card) != 0:
            image = cv.imdecode(np.fromfile(self.dir_card, dtype=np.uint8), -1)
            return image

def im_np2im_tk(im):
    # 改变三通道排列顺序并将图像转换为可显示的类型
    if len(im.shape) == 3:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    img = Image.fromarray(im)
    imTk = ImageTk.PhotoImage(image=img)
    return imTk