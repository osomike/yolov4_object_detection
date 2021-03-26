import sys
sys.path.append('../')

import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
from starlette.responses import StreamingResponse

import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

from app.tools import draw_boxes


object_detection_server = FastAPI()

@object_detection_server.post('/detect_object')
def detect_object(binary_file: UploadFile = File(...)):
    
    # read received file
    my_data = binary_file.file.read()

    # convert binary into numpy array
    my_array = np.frombuffer(my_data, np.uint8)

    # Decode numpy flatten array into cv2 array
    my_img = cv2.imdecode(my_array, cv2.IMREAD_COLOR)

    # Resize image to match YOLOv4 input
    image_resized = tf.image.resize(my_img, (HEIGHT_CNN, WIDTH_CNN))

    # Expand array dimension by 1 axis and nomarlize the array (/255)
    frame = tf.expand_dims(tf.cast(image_resized, tf.float32), axis=0) / 255.0

    # model predictions
    boxes, scores, classes, valid_detections = model.predict(frame)

    # create text fields containing label and score
    labels = ['{} {:.2%}'.format(CLASSES[classes[0].astype(int)[_]], scores.round(3)[0][_]) for _ in range(classes.shape[1])]

    # Display the resulting frame
    new_img = draw_boxes(
        img=my_img,
        rec_coordinates=boxes[0],
        labels=labels,
        colors=None,
        relative_coordinates=True,
        rec_thickness=3,
        label_font_scale=0.7,
        label_font_thickness=1)

    # Save array as image
    #cv2.imwrite('uploaded_img.jpg', my_img)

    return StreamingResponse(io.BytesIO(new_img.tobytes()), media_type="image/jpg")


@object_detection_server.post('/one_way')
def detect_object(binary_file: UploadFile = File(...)):
    
    # read received file
    my_data = binary_file.file.read()

    # convert binary into numpy array
    my_array = np.frombuffer(my_data, np.uint8)

    # Decode numpy flatten array into cv2 array
    my_img = cv2.imdecode(my_array, cv2.IMREAD_COLOR)

    # Resize image to match YOLOv4 input
    image_resized = tf.image.resize(my_img, (HEIGHT_CNN, WIDTH_CNN))

    # Expand array dimension by 1 axis and nomarlize the array (/255)
    frame = tf.expand_dims(tf.cast(image_resized, tf.float32), axis=0) / 255.0

    # model predictions
    boxes, scores, classes, valid_detections = model.predict(frame)

    # create text fields containing label and score
    labels = ['{} {:.2%}'.format(CLASSES[classes[0].astype(int)[_]], scores.round(3)[0][_]) for _ in range(classes.shape[1])]

    # Display the resulting frame
    new_img = draw_boxes(
        img=my_img,
        rec_coordinates=boxes[0],
        labels=labels,
        colors=None,
        relative_coordinates=True,
        rec_thickness=3,
        label_font_scale=0.7,
        label_font_thickness=1)

    # Save array as image
    #cv2.imwrite('uploaded_img.jpg', my_img)

    return {'Detected objects': labels}


@object_detection_server.post('/return_only_box_arrays')
def detect_object(binary_file: UploadFile = File(...)):
    
    # read received file
    my_data = binary_file.file.read()

    # convert binary into numpy array
    my_array = np.frombuffer(my_data, np.uint8)

    # Decode numpy flatten array into cv2 array
    my_img = cv2.imdecode(my_array, cv2.IMREAD_COLOR)

    # Resize image to match YOLOv4 input
    image_resized = tf.image.resize(my_img, (HEIGHT_CNN, WIDTH_CNN))

    # Expand array dimension by 1 axis and nomarlize the array (/255)
    frame = tf.expand_dims(tf.cast(image_resized, tf.float32), axis=0) / 255.0

    # model predictions
    boxes, scores, classes, valid_detections = model.predict(frame)

    # create text fields containing label and score
    labels = ['{} {:.2%}'.format(CLASSES[classes[0].astype(int)[_]], scores.round(3)[0][_]) for _ in range(classes.shape[1])]

    # convert numpy array into a list and then into JSON string
    boxes_json_str = json.dumps(boxes[0].tolist())

    # Return response
    return JSONResponse({'Boxes': boxes_json_str, 'Labels': labels})


@object_detection_server.post('/mirrow')
def detect_object(binary_file: UploadFile = File(...)):
    
    # read received file
    my_data = binary_file.file.read()

    # convert binary into numpy array
    my_array = np.frombuffer(my_data, np.uint8)

    # Decode numpy flatten array into cv2 array
    my_img = cv2.imdecode(my_array, cv2.IMREAD_COLOR)

    # Save array as image
    #cv2.imwrite('uploaded_img.jpg', my_img)

    return StreamingResponse(io.BytesIO(my_img.tobytes()), media_type="image/jpg")


def initialize():

    global WIDTH_CNN, HEIGHT_CNN, model, CLASSES
    WIDTH_CNN, HEIGHT_CNN =  32 * 8, 32 * 6 # (Good enough)

    max_objects = 20

    model = YOLOv4(
        input_shape=(HEIGHT_CNN, WIDTH_CNN, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=max_objects,
        yolo_iou_threshold=0.7,
        yolo_score_threshold=0.7,
    )

    model.load_weights("../binaries/yolov4.h5")

    # COCO classes. Ref https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    print('Initialization done')

initialize()


