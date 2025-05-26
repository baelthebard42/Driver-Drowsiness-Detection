"""
Running this script results in generation of dataset annotations
from the defined dataset path DATASET_PATH. Each key-value pair 
in the annotation represents a single image, with the 'image_path'
key representing the image's full path from main directory while
'targets' represents a one hot encoding of the four output classes.

"""


import json
import os
import copy

DATASET_PATH = "./train"
JSON_PATH = "dataset_annotations.json"
ID_COUNT = 0

TARGETS_MAPPINGS = {
    'yawn_faces':0,
    'no_yawn_faces':1,
    'Closed':2,
    'Open':3
}

DATA_STRUCTURE = { 
    'image_path':None,
    'targets':{
        0:0,
        1:0,
        2:0,
        3:0
    }
}

data = {

}


for dirpath, dirnames, filenames in os.walk(DATASET_PATH):

    if dirpath != DATASET_PATH:
        dirpath_components = os.path.normpath(dirpath).split(os.sep)
        semantic_label = dirpath_components[-1]
      
        print(f"Processing {semantic_label}\n\n")
        
        for f in filenames:

            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
             continue

            img_path = os.path.join(dirpath, f)

            data_point = copy.deepcopy(DATA_STRUCTURE)

            data_point['image_path'] = img_path
            data_point['targets'][TARGETS_MAPPINGS[semantic_label]] = 1

            data[ID_COUNT] = data_point
            ID_COUNT+=1

with open(JSON_PATH, "w") as fp:
    json.dump(data, fp, indent=4)

print(f"Annotations saved as {JSON_PATH}.\nTotal data points obtained: {ID_COUNT}")





