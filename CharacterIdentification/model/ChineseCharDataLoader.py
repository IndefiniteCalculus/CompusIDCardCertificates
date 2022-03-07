import torch
import os
import cv2 as cv
from torchvision import transforms
from CharacterIdentification import ConfigReader as conf
from torch.utils.data import Dataset, DataLoader
from CharacterIdentification import LabelReader
# class CharData(Dataset):
#     def __init__(self, root=conf.get_dir_Chinese_Characters() +"\\train",transform=None):
#         self.labels = LabelReader.get_labels()
#         self.root = root
#         self.tranform = transform
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image_name = os.path.join(self.root, )

def load_data():
    dir,_ = conf.get_dir_Chinese_Characters()
    dir = dir + "\\train"
    files = os.listdir(dir)
    images = []
    for file in files:
        idx = 0
        typefaces = os.listdir(dir +"\\"+ file)
        types = []
        for typeface in typefaces:
            im = cv.imread(dir + "\\" + file + "\\" + typeface)
            im = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
            _,im = cv.threshold(im,0,255, cv.THRESH_OTSU)
            im = 255 - im
            im = cv.resize(im, (40,40))
            # cv.imshow("idx", im)
            # cv.waitKey(500)
            im = im.reshape((1,40,40))
            types.append(im)
        images.append(types)
        idx += 1
    return images

class Chinese_char_dataset(Dataset):
    def __init__(self):
        super(Chinese_char_dataset, self).__init__()
        raw_image = load_data()
        raw_labels = LabelReader.encoded_idx_label(len(raw_image))
        self.images = []
        self.labels = []
        for char_idx in range(len(raw_image)):
            label = raw_labels[char_idx]
            type_faces = raw_image[char_idx]
            for typeface_idx in range(len(type_faces)):
                self.images.append(type_faces[typeface_idx])
                self.labels.append(label)
        self.transform = transforms

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return  image, label

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    test_dataset = Chinese_char_dataset()
    print("dataset created")
    loader = DataLoader(test_dataset, batch_size = 10, shuffle=True)
    for i,data in enumerate(loader):
        images, labels = data[0], data[1].float()
        for idx in range(len(images)):
            im = images[idx].reshape(40,40).data.numpy()
            l = labels[idx].data.numpy()
            cv.imshow(str(idx), im)
            print(LabelReader.decode_label(l))
            cv.waitKey(1000)
            cv.destroyWindow(str(idx))
        pass
