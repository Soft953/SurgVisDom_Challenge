import os
import glob
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SurgVisDataset(Dataset):
    def __init__(self, path, transform=None, img_loader=Image.open, verbose=True):
        self.path = path
        self.img_loader = img_loader
        self.classes = self._get_classes()

        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.X, self.y = self._load_data_path_per_classes()

        self.verbose = verbose

        if self.verbose:
            print(self.classes)
            print(len(self.X), self.y)

    def _get_classes(self):
        # search for subdirectories
        subdir = next(os.walk(self.path))[1] #[x[0] for x in os.walk(self.path)][1:]
        classes = [os.path.basename(os.path.normpath(x)) for x in subdir]
        return dict(zip(classes, range(len(classes))))

    def _load_data_path_per_classes(self):
        X, y = [], []
        for key, value in self.classes.items():
            class_dir = os.path.join(self.path, key)
            extracted_imgs_folder = next(os.walk(class_dir))[1]
            for extracted_folder in extracted_imgs_folder:
                files_path = [os.path.join(class_dir, extracted_folder, x) for x in os.listdir(os.path.join(class_dir, extracted_folder))]
                #import pdb; pdb.set_trace()
                X.append(files_path)
                y.append(value)
        return X, y

    def _preprocess_frame(self, img):
        #Apply transformation (Conversion, Data augmentation, ...)
        return self.transform(img)

    def __getitem__(self, index):
        #load data at idx: index
        index = index % len(self.X)
        frame_idx = np.random.randint(0, len(self.X[index]))

        if self.verbose:
            print(index, frame_idx)
        
        X = self.img_loader(self.X[index][frame_idx])
        y = self.y[index]

        #preprocess frame
        X = self._preprocess_frame(X)
        return X, y

    def __len__(self):
        if hasattr(self, 'length'):
          return self.length
        sum = 0
        for x in self.X:
          sum += len(x)
        self.length = sum
        return self.length

import matplotlib.pyplot as plt

PATH = 'C:\\Users\\gbour\\Desktop\\sysvision\\train_1'
PATH_PORCINE_1 = os.path.join(PATH, 'Porcine')

CROP_SIZE = (420, 630)
INPUT_SHAPE = (256, 256)

train_transform = transforms.Compose([transforms.CenterCrop(CROP_SIZE),
                                        transforms.Resize(INPUT_SHAPE),
                                        transforms.ToTensor()])

dataset = SurgVisDataset(PATH_PORCINE_1, transform=train_transform, verbose=False)
#img, y = dataset.__getitem__(0)

trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))

plt.figure()
plt.imshow(trans(images[0]))
plt.show()
