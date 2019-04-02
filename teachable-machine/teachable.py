#!/usr/bin/env python

# This is a real time learning example, where an already trained model is used for evaluating objects
# and used for re-categorizing your own. This is done through the usage of a method called transfer learning
# https://coral.withgoogle.com/tutorials/edgetpu-retrain-classification-ondevice/


# This code is just an adaptation from the "Teachable machine" example, published in Raspbery PI Magazine #79
# It has been adapted to run within a desktop (a keyboard and a screen is needed), and adapted to run in a
# Linux machine, using CV2 and pygame. For running it in a Raspberry pi, the camera read can be replaced by 
# picamera libraries and if you want to get rid of cv2, the draw_text function can use pygame libraries too.


# run it with with a model with the last fully-connected layer removed (embedding extractor)
# Example:
#      python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite
#
# If you want to save it for later use, you can save the session wby specifying a session name:
#      python3 teachable.py --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite --session my_session1
# (It will create the file containing the labels and the embeddings created during the session)

import argparse
import sys
import os
import time
from collections import deque, Counter
from functools import partial

from embedding import kNNEmbeddingEngine
from PIL import Image

import cv2
import pygame
import numpy as np

class TeachableMachine(object):

  def __init__(self, model_path, kNN=3, buffer_length=4, session=False):
    self._engine = kNNEmbeddingEngine(model_path, kNN, session=session)
    self._buffer = deque(maxlen = buffer_length)
    self._kNN = kNN
    self._start_time = time.time()
    self._frame_times = deque(maxlen=40)
    self.clean_shutdown = False
    self.session_name = session

    BLACK = (0,0,0)
    WIDTH = 800
    HEIGHT = 600
    self.screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

    self.screen.fill(BLACK)

  def classify(self, img, category=None):
        # Classify current image and determine

        emb = self._engine.DetectWithImage(img)
        self._buffer.append(self._engine.kNNEmbedding(emb))
        classification = Counter(self._buffer).most_common(1)[0][0]

        addedMsg = ""
        if category:
            if category == 0 or category == '0': 
                self._engine.clear() # Hitting button 0 resets
            else: 
                self._engine.addEmbedding(emb, category) # otherwise the button # is the class
                addedMsg = "ADDED " + category
            
        self._frame_times.append(time.time())
        fps = len(self._frame_times)/float(self._frame_times[-1] - self._frame_times[0] + 0.001)

        # Print/Display results
        
        status = 'fps: %.1f; #examples: %d; Class: % 7s; '%(
                fps, self._engine.exampleCount(),
                classification or 0) + addedMsg
        #print(status)
        return status

  def saveTrainedModel(self):

    if self.session_name:
        print("Saving...")
        #This sould be on the class or outside here!
        #print(self._engine._embeddings)
        np.save(self.session_name, self._engine._embeddings)
        #print(type(self._engine._labels))
        with open(self.session_name + ".txt", "w") as file:
            file.write(str(self._engine._labels))
        print("Done")
    else:
        print("No session name given. Saving is disabled")
  

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of Tflite model.')
    parser.add_argument('--session', help='The name of the learning data for saving and resuming', default=False)


    args = parser.parse_args()

    print('Initializing Model.')

    teachable = TeachableMachine(args.model, session=args.session)
    pygame.init()

    assert os.path.isfile(args.model)

    cam = cv2.VideoCapture(2) #Change it to zero if you have only one camera
    while True:
            category = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    teachable.saveTrainedModel()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    category = pygame.key.name(event.key)

            #Get the image and transform it into a pil image
            ret, cv2_im = cam.read()
            new_cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(new_cv2_im)

            #Classify the image
            status = teachable.classify(pil_im, category)

            #Draw the results
            draw_text(new_cv2_im, status)
            new_cv2_im = cv2.resize(new_cv2_im, (800,600))
 
            #put back the image into the screen using a pygame image
            pygameimage = pygame.image.frombuffer(new_cv2_im.tostring(), new_cv2_im.shape[1::-1],"RGB")
            teachable.screen.blit(pygameimage, (0,0))
            pygame.display.update()

def draw_text(image_np, label, pos=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    p1 = (0,pos*30+20)
    cv2.rectangle(image_np, (p1[0], p1[1]-20), (800, p1[1]+10) ,color=(0, 255, 0), thickness = -1)
    cv2.putText(image_np, label, p1, font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
