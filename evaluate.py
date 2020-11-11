import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch import load
from torchvision import transforms

from model import Net
from data_loader import SurgVisTestset

def evaluate(testsetloader, net, batch_size):
    #define optimizer and loss function
    #criterion = nn.CrossEntropyLoss()

    model.eval() # Set model to evaluate mode

    predictions = []

    with torch.no_grad():
        #evaluation loop
        for i in range(len(testsetloader.X)):  # loop over each video

            print('Video number:', i, end="")
            start = time.time()
            for j in range(0, testsetloader.metadata.total_num_frames.iloc[i], batch_size):
                images = testsetloader.get_batches(i, list(range(j, j+batch_size)))
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                #c = (predicted == labels).squeeze()
                #for i in range(4):
                #    label = labels[i]
                #    class_correct[label] += c[i].item()
                #    class_total[label] += 1
                predictions += predicted
            end = time.time()
            print(" -- time elapsed (in sec):", end - start)
            break

    print('Finished evaluation')
    return predictions


if __name__ == "__main__":
    BATCH_SIZE = 32
    CROP_SIZE = (420, 630)
    INPUT_SHAPE = (256, 256)

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.CenterCrop(CROP_SIZE),
                                            transforms.Resize(INPUT_SHAPE),
                                            transforms.ToTensor()])

    test_set = SurgVisTestset(path=Path('C:\\Users\\gbour\\Desktop\\sysvision\\test'), transform=train_transform, verbose=False)

    model = Net()
    model.load_state_dict(torch.load(str(Path('C:\\Users\\gbour\\Desktop\\sysvision\\Models\\cnn_first_training.pth')), map_location=torch.device('cpu')))

    predictions = evaluate(test_set, model, BATCH_SIZE)
    pd.DataFrame(predictions).to_csv('test_result.csv', index=False)
