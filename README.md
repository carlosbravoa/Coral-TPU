# Coral-TPU
Examples for playing with the coral edge TPU

### My_TPU_object_recognition.py
My_TPU_object_recognition.py is a demo for object detection or image classification using CORAL TPU, using a coninuous streaming from a camera (different from the base examples, where they just classify static images)

**This example is intended to run later in a raspberry PI, but for now, is running on a Lunux machine. The only pending thing to make it run on the raspberry, since capturing frames require a different method through the picamera python library**
***See: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

For running in a Linux PC, follow the standard installation of the CORAL TPU USB, plus installing Python-OpenCV

Just be sure to have run the CORAL EDGE TPU setup.py. If you just place this file with the rest of the examples, it will run.
