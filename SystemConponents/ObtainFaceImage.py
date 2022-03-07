import cv2 as cv

def get_image(camera):
    ret, frame = camera.read()
    if ret is True:
        return frame

if __name__ == "__main__":
    camera = cv.VideoCapture(0)
    while camera.isOpened():
        frame = get_image(camera)
        cv.imshow('camera',frame)
        cv.waitKey(200)
    pass