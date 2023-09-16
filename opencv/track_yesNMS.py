# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from deep_sort_realtime.deepsort_tracker import DeepSort
# from ds_tracker import DeepSort
import time
from nms import non_max_suppression

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_custom_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    # cap.set(cv2.CAP_PROP_FPS, 20)
    tracker = DeepSort(max_age=30,n_init=2,nms_max_overlap=1,max_cosine_distance=0.3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2_im = frame
        start = time.perf_counter()

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im, tracker_list= append_objs_to_img(cv2_im, inference_size, objs, labels)  

        # tracking algorithm (DeepSort)
        tracks = tracker.update_tracks(tracker_list, frame=cv2_im) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            cv2_im = cv2.rectangle(cv2_im, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 0, 255), 2)
            cv2_im = cv2.putText(cv2_im, "ID: " + str(track_id), (int(ltrb[0]), int(ltrb[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            # cv2_im = cv2.putText(cv2_im, label, (int(ltrb[0]), int(ltrb[1]+30)),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime

        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    tracker_list=[]
    
    interest = [
        'handbag', 'backpack', 'suitcase', 'person'
    ]

    for obj in objs:
        label = labels.get(obj.id, obj.id)
        # if label in interest:
        # Convert the object bounding boxes to the format expected by non_max_suppression
        boxes = np.array([[obj.bbox.scale(scale_x, scale_y).xmin, obj.bbox.scale(scale_x, scale_y).ymin,
                        obj.bbox.scale(scale_x, scale_y).xmax, obj.bbox.scale(scale_x, scale_y).ymax,
                        obj.id, obj.score]])

        # Apply non-maximum suppression to obtain the final list of detection
        if np.size(boxes) != 0:
            scores = boxes[:,5]
            pick = non_max_suppression(boxes, scores, overlapThresh=0.3)
        
            # Loop over the picked indexes and draw the corresponding bounding boxes and labels
            for bbox in pick:
                # each row represent a box
                x0, y0 = int(bbox[0]), int(bbox[1])
                x1, y1 = int(bbox[2]), int(bbox[3])
                score = bbox[6]
                label = '{}% {}'.format(score, labels.get(bbox[4], bbox[4]))
                
                # cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                
                # appending tracked objects coordinates 
                tracker_list.append(([x0, y0, int(x1-x0), int(y1-y0)], score, str(labels.get(bbox[4], bbox[4]))))

    return cv2_im, tracker_list

if __name__ == '__main__':
    main()
