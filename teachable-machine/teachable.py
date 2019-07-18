#!/usr/bin/env python

'''

***WARNING: SPAGHETTI CODE AHEAD... USE IT AT YOUR OWN RISK ***

This is a real time learning example, where an already trained model is used for
evaluating objects and used for re-categorizing your own. This is done through the
usage of a method called transfer learning
https://coral.withgoogle.com/tutorials/edgetpu-retrain-classification-ondevice/


This code is just an adaptation from the "Teachable machine" example, published in
Raspbery PI Magazine #79.
It has been adapted to run within a desktop (a keyboard and a screen is needed),
and adapted to run in a Linux machine, using CV2 and pygame. For running it in a
Raspberry pi, the camera read can be replaced by picamera libraries and if you want
to get rid of cv2, the draw_text function can use pygame libraries too.


run it with with a model with the last fully-connected layer removed (embedding extractor)

Example:
      python3 teachable.py \
        --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite

If you want to save it for later use, you can save the session wby specifying a session name:
      python3 teachable.py \
        --model downloaded-models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite \
        --session my_session1
 (It wipygame ll create the file containing the labels and the embeddings created during the session)

'''
import argparse
import sys
import os
import time

import cv2
import pygame
import numpy as np

from collections import deque, Counter
from functools import partial
from PIL import Image

from embedding import kNNEmbeddingEngine



class TeachableMachine(object):

    def __init__(self, model_path, kNN=3, buffer_length=4, session=False):
        self._engine = kNNEmbeddingEngine(model_path, kNN, session=session)
        self._buffer = deque(maxlen=buffer_length)
        self._kNN = kNN
        self._start_time = time.time()
        self._frame_times = deque(maxlen=40)
        self.clean_shutdown = False
        self.session_name = session

        self.categoriesImageDic = dict()

        BLACK = (0, 0, 0)
        WIDTH = 800
        HEIGHT = 600
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

        self.screen.fill(BLACK)

    def classify(self, pil_img):

        # Classify current image
        emb = self._engine.DetectWithImage(pil_img)
        self._buffer.append(self._engine.kNNEmbedding(emb))
        classification = Counter(self._buffer).most_common(1)[0][0]

        return classification

    def addCategory(self, pil_img, np_img, category=None):
        if category:
            if category == 0 or category == '0':
                self._engine.clear() # Hitting button 0 resets
            else:
                emb = self._engine.DetectWithImage(pil_img)
                self._engine.addEmbedding(emb, category) 
                if not category in self.categoriesImageDic:
                    self.categoriesImageDic[category] = cv2.resize(np_img, (0,0), fx=0.3, fy=0.3)
                    #displayThumbnail(np_img, self.categoriesImageDic[category])

    def save_trained_model(self):

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
    parser.add_argument('--session',
                        help='The name of the learning data for saving and resuming',
                        default=False)

    args = parser.parse_args()

    print('Initializing Model.')

    teachable = TeachableMachine(args.model, session=args.session)
    pygame.init()

    assert os.path.isfile(args.model)

    cam = cv2.VideoCapture(2) #Change it to zero if you have only one camera
    while True:
        newCategory = None

        # Listening for keyboard/UI events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                teachable.save_trained_model()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                newCategory = pygame.key.name(event.key)

        # Get the image  from the camera
        ret, cv2_im = cam.read()

        # The engine uses pil images as input, so, we need to transform it
        cv2RGB = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2RGB)

        statusMessage  = ""

        # If there is a new category, add it with the image
        if newCategory:
            teachable.addCategory(pil_img, cv2_im, newCategory)
            statusMessage = "ADDED " + newCategory + " category"

        # Else, classify the image
        else: 
            classification = teachable.classify(pil_img)

            # And display the thumbnail
            if classification:
                displayThumbnail(
                            cv2_im, 
                            teachable.categoriesImageDic[classification]
                            )

        # For getting the FPS and display
        teachable._frame_times.append(time.time())
        fps = len(teachable._frame_times)/float(teachable._frame_times[-1] - teachable._frame_times[0] + 0.001)

        status = 'fps: %.1f; #examples: %d; Class: % 7s; '%(
            fps, teachable._engine.exampleCount(),
            classification or 0) + statusMessage

        draw_text(cv2_im, status)
        #new_cv2_im = cv2.resize(new_cv2_im, (800, 600))

        #put back the image into the screen using a pygame image
        cv2RGB = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pygameimage = pygame.image.frombuffer(
            cv2RGB.tostring(), cv2RGB.shape[1::-1], "RGB")
        teachable.screen.blit(pygameimage, (0, 0))
        pygame.display.update()

def draw_text(image_np, label, pos=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    p1 = (0, pos*30+20)
    cv2.rectangle(image_np, (p1[0], p1[1]-20), (800, p1[1]+10), color=(0, 255, 0), thickness=-1)
    cv2.putText(image_np, label, p1, font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

def displayThumbnail(background, overlay):
    rows,cols,channels = overlay.shape
    brows, bcols, channels = background.shape
    #overlay=cv2.addWeighted(background[250:250+rows, 0:0+cols],0.5,overlay,0.5,0) @if transparency is needed
    #    background[250:250+rows, 0:0+cols ] = overlay
    background[brows-rows:brows, bcols-cols:bcols ] = overlay

if __name__ == '__main__':
    sys.exit(main(sys.argv))
