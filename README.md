# Coral-TPU demo
Examples for playing with the [coral edge TPU accelerator](https://coral.withgoogle.com/tutorials/accelerator/).

## My_TPU_object_recognition.py
My_TPU_object_recognition.py is a demo for object detection or image classification using CORAL TPU, using a coninuous streaming from a camera (different from the base examples, where they just classify static images)

**This example runs on a raspberry PI, but using an usb camera. The only pending thing to make it run on the raspberry picamera,
the difference is that here we are using cv2 and capturing frames through picam requirse a different method through the picamera
python library**

***See: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

For running in a Linux PC, follow the standard installation of the CORAL TPU USB, plus installing Python-OpenCV

Just be sure to have run the CORAL EDGE TPU setup.py.

**There is a second example called MY_TPU_object_recognition2.py that is using pygame for displaying the video. 
I'm only getting an small fps improvement, but not enough**

You need to have downloaded somewhere the edgecpu tflite models and the labels.
Example to run it (from desktop to see the results in real-time): 

  - Object recognition:
    python3 ./my_TPU_image_recognition2.py \
    --model=models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --label=models/coco_labels.txt --mode=OBJECT_DETECTION \
    --camera=0



# The teachable machine

This is a real time learning example, where an already trained model is used for evaluating objects and used for re-categorizing your own. This is done through the usage of a method called transfer learning https://coral.withgoogle.com/tutorials/edgetpu-retrain-classification-ondevice/

This code is just an adaptation from the "Teachable machine" example, published in Raspbery PI Magazine #79. It has been adapted to run within a desktop (a keyboard and a screen is needed), and adapted to run in a Linux machine, using CV2 and pygame. For running it in a Raspberry pi, the camera read can be replaced by picamera libraries and if you want to get rid of cv2, the draw_text function can use pygame libraries too.


run it with with a model with the last fully-connected layer removed (embedding extractor)

Example:
`python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`

If you want to save it for later use, you can save the session wby specifying a session name:
      `python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite --session my_session1`
(It will create the file containing the labels and the embeddings created during the session)

Can run standalone if you have already installed all Coral TPU libraries.
