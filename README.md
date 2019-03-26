# Coral-TPU
Examples for playing with the coral edge TPU

### My_TPU_object_recognition.py
My_TPU_object_recognition.py is a demo for object detection or image classification using CORAL TPU, using a coninuous streaming from a camera (different from the base examples, where they just classify static images)

**This example is intended to run later in a raspberry PI, but for now, is running on a Lunux machine. The only pending thing to make it run on the raspberry, since capturing frames require a different method through the picamera python library**
***See: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

For running in a Linux PC, follow the standard installation of the CORAL TPU USB, plus installing Python-OpenCV

Just be sure to have run the CORAL EDGE TPU setup.py. If you just place this file with the rest of the examples, it will run.

# The teachable machine

This is a real time learning example, where an already trained model is used for evaluating objects and used for re-categorizing your own. This is done through the usage of a method called transfer learning https://coral.withgoogle.com/tutorials/edgetpu-retrain-classification-ondevice/

This code is just an adaptation from the "Teachable machine" example, published in Raspbery PI Magazine #79. It has been adapted to run within a desktop (a keyboard and a screen is needed), and adapted to run in a Linux machine, using CV2 and pygame. For running it in a Raspberry pi, the camera read can be replaced by picamera libraries and if you want to get rid of cv2, the draw_text function can use pygame libraries too.


run it with with a model with the last fully-connected layer removed (embedding extractor)
Example:
`python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`

