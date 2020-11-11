import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torch import stack
from torch.utils.data import Dataset, DataLoader, random_split
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
                X.append(files_path)
                y.append(value)
        return X, y

    def _preprocess_frame(self, img):
        #Apply transformation (Conversion, Data augmentation, ...)
        return self.transform(img)

    def __getitem__(self, index):
        #load data at idx: index
        index_1 = index % len(self.X)
        frame_idx = np.random.randint(0, len(self.X[index_1]))

        if self.verbose:
            print(index, index_1, frame_idx)
        
        X = self.img_loader(self.X[index_1][frame_idx])
        y = self.y[index_1]

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


class SurgVisTestset:
    """
        Simple Custom Dataset because test set is too big to be saved and loaded frame by frame
    """
    def __init__(self, path, transform=None, verbose=True):
        self.path = Path(path)

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.metadata = pd.read_csv(self.path.joinpath('testing_set_video_metadata.csv'))
        self.X = [[] for i in range(len(self.metadata))]

        self.verbose = verbose

        if self.verbose:
            print(self.metadata.head())
            print(len(self.X))

    def _extract_frames_from_videos(self, vid_path : Path, frame_batch):
        cap = cv2.VideoCapture(str(vid_path))

        frames = []
        for idx in frame_batch:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            frames.append(frame)
        
        return frames

    def _load_frames(self, vid_index, frame_batch):
        vid_path = self.path.joinpath(self.metadata.vid_name.iloc[vid_index] + '.mp4')
        return self._extract_frames_from_videos(vid_path, frame_batch)

    def get_batches(self, vid_index, indexes):
        #print(indexes)
        frames = self._load_frames(vid_index, indexes)
        list_of_tensors = []
        for frame in frames:
            if frame is None:
                # Skip the last frame
                continue
            #print(type(frame), frame.shape)
            frame_as_tensor = self.transform(frame)
            list_of_tensors.append(frame_as_tensor)
        return stack(list_of_tensors)

CROP_SIZE = (420, 630)
INPUT_SHAPE = (256, 256)

train_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.CenterCrop(CROP_SIZE),
                                        transforms.Resize(INPUT_SHAPE),
                                        transforms.ToTensor()
                                    ])

test_set = SurgVisTestset(path=Path('C:\\Users\\gbour\\Desktop\\sysvision\\test'), transform=train_transform, verbose=True)
batches = test_set.get_batches(0, list(range(32)))
print(batches.shape)


import matplotlib.pyplot as plt

PATH = 'C:\\Users\\gbour\\Desktop\\sysvision\\train_1'
PATH_PORCINE_1 = os.path.join(PATH, 'Porcine')



train_transform = transforms.Compose([transforms.CenterCrop(CROP_SIZE),
                                        transforms.Resize(INPUT_SHAPE),
                                        transforms.ToTensor()])

dataset = SurgVisDataset(PATH_PORCINE_1, transform=train_transform, verbose=True)
"""
#img, y = dataset.__getitem__(0)
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))



n = len(dataset)
n_train = int(n * 0.9)
n_val = int(n * 0.1)
print("dataset len:", n)


train_set, val_set, test_set = random_split(dataset, (n_train, n_val, 0))

print("Train set:", len(train_set))
print("Val set:", len(val_set))
print("Test set", len(test_set))


import pdb; pdb.set_trace()


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))
"""
"""
plt.figure()
plt.imshow(trans(images[0]))
plt.show()
"""