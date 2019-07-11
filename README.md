# Coral-TPU demo
(https://github.com/carlosbravoa/Coral-TPU)

Examples for playing with the [coral edge TPU accelerator](https://coral.withgoogle.com/tutorials/accelerator/).

**Update July 2019. It works smoothly in Raspberry pi4. There is a small modification needed in the edgeTPU installation script and the steps needed for linking the edge_tpu lib**

These examples are a demonstration for object recognition and image classification in real time, using CORAL EDGE TPU, using a coninuous streaming from a camera (different from the base examples, where they just classify static images) and drawing the results in realtime on the screen.

These examples run both on Linux PC and Raspberry-pi (there is an example that runs with raspberry pi camera only. See below)

## My_TPU_object_recognition.py
My_TPU_object_recognition.py is a demo for object detection or image classification using CORAL EDGE TPU, using cv2 for capturing and anotating the image with the results.

**This example runs also on a raspberry PI, using an usb camera.**

For running it, follow the standard installation of the [CORAL EDGE TPU USB](https://coral.withgoogle.com/docs/accelerator/get-started/), plus installing Python-OpenCV (for Raspberry Pi, it may need compiling!)

Just be sure to have run and set up the [CORAL EDGE TPU USB](https://coral.withgoogle.com/docs/accelerator/get-started/) setup.py.

**There is a second example called MY_TPU_object_recognition2.py that is using pygame for displaying the video.**

**A third example, called my_TPU_image_recognition_picam.py is available for using the raspberry pi camera** There is a better example (better performing with no dependencies with CV2 [here](https://github.com/nickoala/edgetpu-on-pi)

You need to have downloaded the edgecpu tflite models and the labels from here: [Pre compiled models for edge TPU](https://coral.withgoogle.com/models/).

Example for runnining any of these 3 scripts (It has to be executed from desktop, since the results are displayed in real-time): 

  - Object detection:
    python3 ./my_TPU_image_recognition.py \
    --model=models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --label=models/coco_labels.txt --mode=OBJECT_DETECTION \
    --camera=0

  - Image classification:
    python3 ./my_TPU_image_recognition.py \
    --model=models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --label=models/coco_labels.txt --mode=IMAGE_CLASSIFICATION \
    --camera=0


## The teachable machine

This is a real time learning example, where an already trained model is used for evaluating objects and used for re-categorizing your own. This is done through the usage of a method [called transfer learning](https://coral.withgoogle.com/tutorials/edgetpu-retrain-classification-ondevice/).

This code is just an adaptation from the "Teachable machine" example, published in Raspbery PI Magazine #79. It has been adapted to run within a desktop (a keyboard and a screen is needed), and adapted to run in a Linux machine (PC or Raspberry Pi), using CV2 and pygame. For running it in a Raspberry pi, the camera read can be replaced by picamera libraries and if you want to get rid of cv2, as shown in the picam example above.


Run it with with a model with the last fully-connected layer removed (embedding extractor)

Example:
`python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`

If you want to save it for later use, you can save the session wby specifying a session name:
      `python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite --session my_session1`
(It will create the file containing the labels and the embeddings created during the session)

Can run standalone if you have already installed all Coral TPU libraries.
