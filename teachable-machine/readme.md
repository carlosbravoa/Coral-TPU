# The teachable machine

This is a real time learning example, where an already trained model is used for evaluating objects and used for re-categorizing your own. This is done through the usage of a method called transfer learning: https://coral.withgoogle.com/tutorials/edgetpu-retrain-classification-ondevice/


This code is just an adaptation from the "Teachable machine" example, published in Raspbery PI Magazine #79
It has been adapted to run within a desktop (a keyboard and a screen is needed), and adapted to run in a
Linux machine, using CV2 and pygame. For running it in a Raspberry pi, the camera read can be replaced by 
picamera libraries and if you want to get rid of cv2, the draw_text function can use pygame libraries too.


Run it with with a model with the last fully-connected layer removed (embedding extractor)

Example:
`python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`

If you want to save it for later use, you can save the session wby specifying a session name:
      `python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite --session my_session1`
(It will create the file containing the labels and the embeddings created during the session)


