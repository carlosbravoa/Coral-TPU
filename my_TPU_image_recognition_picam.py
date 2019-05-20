"""A demo for object detection or image classification using CORAL TPU.

This example is intended to run later in a raspberry PI, but for now, is running on a
Linux machine

The only pending thing to make it run on the raspberry, since capturing frames require
a different method through the picamera python library
See:
https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python

For running in a Linux PC, follow the standard installation of the CORAL TPU USB, plus
installing Python-OpenCV

Examples (Running under python-tflite-source/edgetpu directory):
  - Object recognition:
    python3 demo/my_TPU_image_recognition.py \
    --model=test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --label=test_data/coco_labels.txt --mode=OBJECT_DETECTION \
    --camera=0

  - Image classification (plants from iNat):
    python3 demo/my_TPU_image_recognition.py \
    --model=test_data/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite \
    --label=test_data/inat_plant_labels.txt  --mode=IMAGE_CLASSIFICATION

  - Image classification (InceptionV4 ImageNet)
    python3 demo/my_TPU_image_recognition.py \
    --model test_data/inception_v4_299_quant_edgetpu.tflite \
    --label=test_data/imagenet_labels.txt --mode=IMAGE_CLASSIFICATION

  - Face detection:
    python3 demo/object_detection.py \
    --model='test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite' \
    --mode=IMAGE_CLASSIFICATION'

  - Pet detection:
    python3 demo/object_detection.py \
    --model='test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite' \
    --label='test_data/pet_labels.txt' \

"""
import argparse
import platform
import subprocess
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
import numpy as np
import time
from collections import deque, Counter

#For webcam capture and drawing boxes
import cv2

#picamera
from picamera import PiCamera
import io

# Parameters for visualizing the labels and boxes
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.7
LABEL_BOX_PADDING = 5
LABEL_BOX_OFFSET_TOP = int(20 * FONT_SIZE) + LABEL_BOX_PADDING
LINE_WEIGHT = 1

# Function to read labels from text files.
def read_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='Path of the detection model.', required=True)
    parser.add_argument(
        '--label', help='Path of the labels file.')
    parser.add_argument(
        '--mode', help='Mode for de detection: OBJECT_DETECTION or IMAGE_CLASSIFICATION',
        required=True)
    parser.add_argument(
        '--camera', help='Camera source (if multiple available)', type=int, required=False)

    args = parser.parse_args()

    # Initialize engine.

    if args.mode == "OBJECT_DETECTION":
        engine = DetectionEngine(args.model)
    elif args.mode == "IMAGE_CLASSIFICATION":
        engine = ClassificationEngine(args.model)
    else:
        print("Please insert the mode from OBJECT_DETECTION or IMAGE_CLASSIFICATION")
        exit()

    labels = read_label_file(args.label) if args.label else None
    label = None
    camera = args.camera if args.camera else 0

    # Initialize the camera
    #cam = cv2.VideoCapture(camera)
    camera = PiCamera()
    time.sleep(2)
    # Create the in-memory stream
    stream = io.BytesIO()

    # Initialize the timer for fps
    start_time = time.time()
    frame_times = deque(maxlen=40)

    while True:
        #ret, cv2_im = cam.read()
        camera.capture(stream, format='jpeg')
        stream.seek(0)
        pil_im = Image.open(stream)
        cv2_im = np.array(pil_im)
        cv2_im = cv2.cvtColor(cv2_im, cv2.cv.CV_BGR2RGB)
        #we are transforming the npimage to img, and the TPU library/utils are doing the
        #inverse process
        #The CV2 Way
        #pil_im = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB))
        #pil_im = Image.fromarray(np.uint8(cv2_im)).convert('RGB') 
        #This is the tf utils way for the transformation. It needs numpy, and is slightly slower
        
        
        if args.mode == "OBJECT_DETECTION":
            ans = engine.DetectWithImage(pil_im, threshold=0.05, keep_aspect_ratio=True,
                                         relative_coord=False, top_k=10)
            if ans:
                print("{} object(s) detected".format(len(ans)))
                for obj in ans:
                    if obj.score > 0.4:
                        if labels:
                            label = labels[obj.label_id] + " - {0:.2f}".format(obj.score)
                        draw_rectangles(obj.bounding_box, cv2_im, label=label)
            else:
                draw_text(cv2_im, 'No object detected!')

        else:
            i = 0
            for result in engine.ClassifyWithImage(pil_im, top_k=5):
                if result:
                    label = labels[result[0]]
                    score = result[1]

                    draw_text(cv2_im, label, i)
                    i += 1
                else:
                    draw_text(cv2_im, 'No classification detected!')
        lastInferenceTime = engine.get_inference_time()
        frame_times.append(time.time())
        fps = len(frame_times)/float(frame_times[-1] - frame_times[0] + 0.001)
        draw_text(cv2_im, "{:.1f} / {:.2f}ms".format(fps, lastInferenceTime))


        #flipping the image: cv2.flip(cv2_im, 1)
	
	#cv2_im = cv2.resize(cv2_im, (800, 600))
        cv2.imshow('object detection', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
            break

    #end
    #cv2.VideoCapture.release(cam)

def draw_rectangles(rectangles, image_np, label=None):
    p1 = (int(rectangles[0][0]), int(rectangles[0][1]))
    p2 = (int(rectangles[1][0]), int(rectangles[1][1]))
    cv2.rectangle(image_np, p1, p2, color=(255, 0, 0), thickness=LINE_WEIGHT)
    if label:
        cv2.rectangle(image_np, (p1[0], p1[1]-LABEL_BOX_OFFSET_TOP), (p2[0], p1[1] + LABEL_BOX_PADDING),
                      color=(255, 0, 0),
                      thickness=-1)
        cv2.putText(image_np, label, p1, FONT, FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)
    #imgname = str(time.time())
    #cv2.imwrite('/home/pi/development/Coral-TPU/imgs/' + imgname + '.jpg', image_np)

def draw_text(image_np, label, pos=0):
    p1 = (0, pos*30+20)
    #cv2.rectangle(image_np, (p1[0], p1[1]-20), (800, p1[1]+10), color=(0, 255, 0), thickness=-1)
    cv2.putText(image_np, label, p1, FONT, FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == '__main__':
    main()
