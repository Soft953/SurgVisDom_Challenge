import os
from pathlib import Path
import cv2

def _extract_imgs_from_videos(vid_path):
    cap = cv2.VideoCapture(vid_path)
    i = 0
    vid_name = os.path.basename(os.path.normpath(vid_path))
    path = vid_path[:-len(vid_name)]
    print("Path:", path, "-- Vid Name:", vid_name)
    imgs_folder = os.path.join(path, vid_name.split('.')[0])
    try:
        os.mkdir(imgs_folder)
    except:
        print("folder already exist!")
        return

    while(cap.isOpened()): 
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imwrite(os.path.join(imgs_folder, "frame_" + str(i) + ".png"), frame)
        i += 1
        #cv2.resize(frame[60:490, 165:795], (256, 256))))

def _load_data_per_classes(path):
    PATH_PORCINE_1 = os.path.join(path, 'Porcine')
    path = PATH_PORCINE_1
    for key, value in {'Dissection': 0, 'Knot_Tying': 1, 'Needle_Driving': 2}.items():
        class_dir = os.path.join(path, key)
        print(class_dir)
        files_path = [os.path.join(class_dir, x) for x in os.listdir(class_dir)]
        print(files_path)
        for f in files_path:
            _extract_imgs_from_videos(f)

if __name__ == '__main__':
    # PATH = 'C:\\Users\\gbour\\Desktop\\sysvision\\train_1'
    PATH = Path('C:\\Users\\gbour\\Desktop\\sysvision\\test')
    if PATH.name == "test":
        for vid_path in PATH.iterdir():
            #vid_name = vid_path.name.split('.')[0]
            #PATH.joinpath(vid_name).mkdir()
            #print(PATH.joinpath(vid_name))
            _extract_imgs_from_videos(str(vid_path))
    else:
        _load_data_per_classes(PATH)

