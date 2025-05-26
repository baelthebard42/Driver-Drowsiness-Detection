"""
Running this script results in the images in the specified BASE_DIR and SUB_DIRS 
directories to be resized to specified RESIZE_TO
"""


import cv2, os

BASE_DIR = './train'
SUB_DIRS = ['Closed', 'Open']
RESIZE_TO = 145

for dir in SUB_DIRS:
    print(f"\nResizing dir {dir}")

    for image in os.listdir(os.path.join(BASE_DIR, dir)):
        img = cv2.imread(os.path.join(BASE_DIR, dir, image))
        cv2.imwrite(os.path.join(BASE_DIR, dir, image), cv2.resize(img, (RESIZE_TO, RESIZE_TO)))
    print(f"\nResizing done")