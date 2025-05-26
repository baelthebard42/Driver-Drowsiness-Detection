"""
Running this script results in the extraction and saving of faces 
 from images in specified directories BASE_DIR and SUB_DIRS to the SAVE_DIRS with
 SAVE_IMG_SIZE.



"""

import cv2, os
import mediapipe as mp
from mediapipe.python.solutions.face_detection import FaceDetection

BASE_DIR = './train'
SUB_DIRS = ['yawn', 'no_yawn']
SAVE_DIRS = ['yawn_faces', 'no_yawn_faces']
SAVE_IMG_SIZE = 145


def extract_cropped_faces(img_path, model:FaceDetection):

      img = cv2.imread(img_path)
      h_img, w_img, _ = img.shape
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      result = model.process(gray)

      if not result.detections:
         print(f"face not detected for {img_path}")
         return None
      
      bbox =result.detections[0].location_data.relative_bounding_box
      x = int(bbox.xmin * w_img)
      y =  int(bbox.ymin * h_img)
      w = int(bbox.width * w_img)
      h = int(bbox.height * h_img)
      
      cropped_face = img[y:y+h, x:x+w]
      
      return cropped_face


mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5 ) as face_detection:

  for idx, dir in enumerate(SUB_DIRS):
 
   save_path = os.path.join(BASE_DIR, SAVE_DIRS[idx])

   if not os.path.exists(save_path):
    os.makedirs(save_path)
 
   current_dir = os.path.join(BASE_DIR, dir)
   print(f"\n\nWorking on directory {current_dir}")

   for image in os.listdir(current_dir):
    face = extract_cropped_faces(os.path.join(current_dir, image), face_detection)

    if face is not None and face.size!=0:
     cv2.imwrite(os.path.join(save_path, image), cv2.resize(face, (SAVE_IMG_SIZE, SAVE_IMG_SIZE)))
    else:
     print(f"Problem with {os.path.join(current_dir, image)}")
  
  print(f"\n\nSaved all extracted faces from {current_dir} to {save_path} ")








