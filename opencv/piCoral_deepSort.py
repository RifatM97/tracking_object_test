from multiprocessing import Process, Value, Queue
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import cv2 
import numpy as np
from ultralytics import YOLO
import time
from deep_sort.application_util import preprocessing

# Initialize deepsort tracker parameters
max_cosine_distance = 0.4
nn_budget = None
model_filename = '../model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, 5, 2)

# initialise yolo detection model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 20)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start = time.perf_counter()
    results = model(frame)
    for result in results: 
        bboxs = []
        scores = []
        names = []

        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, id = r
            box = [int(x1), int(y1), int(x2), int(y2)]
            bboxs.append(box)
            scores.append(score)
            names.append(id)

        names = np.array(names)
        bboxs = np.array(bboxs)
        scores = np.array(scores)
        print("detection list:", bboxs)
        print("len:", len(bboxs))

        # tracking algorithm (DeepSort)
        features = encoder(frame, bboxs)
        detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxs, scores, features)]

        print("deepsort Detections:", detections)

        # grab boxes, scores, and classes_name from deep sort detections
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        #perform non-maxima suppression on deep sort detections
        indices = preprocessing.non_max_suppression(boxs, 0.3, scores)
        detections = [detections[i] for i in indices]

        # DeepSORT -> Predicting Tracks.
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # grab object bounding box class_name
            bbox = track.to_tlbr()
            class_name= track.get_class()

            # calculate centroid of each box
            cx, cy = int((bbox[0] + bbox[2])/2.0), int((bbox[1] + bbox[3])/2.0)
            centroid = (cx, cy)
            
            #draw bbox and track id on frame 
            cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 0, 255), 2)
            y = int(bbox[1]) - 15 if int(bbox[1]) - 15 > 15 else int(bbox[1]) + 15
            cv2.putText(frame, class_name + ' ' + str(track.track_id), (int(bbox[0]), y), 0, 0.5, (255, 0, 0), 2)
            

            end = time.perf_counter()
            totalTime = end - start
            fps = 1 / totalTime
    
            text = 'Average FPS: {:.2f}%'.format(fps)
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
 

        ##########

    #     for track in tracks:
    #         if not track.is_confirmed():
    #             continue
    #         track_id = track.track_id
    #         ltrb = track.to_ltrb()
    

    #         cv2_im = cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 0, 255), 2)
    #         cv2_im = cv2.putText(frame, "ID: " + str(track_id), (int(ltrb[0]), int(ltrb[1]) - 10), 
    #                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            

    # end = time.perf_counter()
    # totalTime = end - start
    # fps = 1 / totalTime

    # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()