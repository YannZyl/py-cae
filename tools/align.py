# -*- coding: utf-8 -*-
#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

modelDir = os.path.join('openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dlibFacePredictor = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
sample_numImages = 0
def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def computeMeanMain(inputDir, outputDir):
    openface.helper.mkdirP(outputDir)
    align = openface.AlignDlib(dlibFacePredictor)

    imgs = list(iterImgs(inputDir))
    if sample_numImages > 0:
        imgs = random.sample(imgs, sample_numImages)

    facePoints = []
    for img in imgs:
        rgb = img.getRGB()
        bb = align.getLargestFaceBoundingBox(rgb)
        if bb:
            alignedPoints = align.findLandmarks(rgb, bb)    
            if alignedPoints:
                facePoints.append(alignedPoints)

    facePointsNp = np.array(facePoints)
    mean = np.mean(facePointsNp, axis=0)
    std = np.std(facePointsNp, axis=0)

    write(mean, "{}/mean.csv".format(dlibModelDir))
    write(std, "{}/std.csv".format(dlibModelDir))

    # Only import in this mode.
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(mean[:, 0], -mean[:, 1], color='k')
    ax.axis('equal')
    for i, p in enumerate(mean):
        ax.annotate(str(i), (p[0] + 0.005, -p[1] + 0.005), fontsize=8)
    plt.savefig("{}/mean.png".format(dlibModelDir))


def align_face(im_size, inputDir, outputDir='data/aligned_dataset'):
    openface.helper.mkdirP(outputDir)

    imgs = list(iterImgs(inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)
    landmarks = 'innerEyesAndBottomLip'
    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    landmarkIndices = landmarkMap[landmarks]

    align = openface.AlignDlib(dlibFacePredictor)
    for imgObject in imgs:
        print "Aligned images: {}".format(imgObject.path)       
        ext = imgObject.path.split('.')[-1]
        outDir = os.path.join(outputDir, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + "." + ext

        rgb = imgObject.getRGB()
        if rgb is None:
            outRgb = None
        else:
            outRgb = align.align(im_size, rgb,
                                 landmarkIndices=landmarkIndices,
                                 skipMulti=True)
   
        if outRgb is None:
            shutil.copyfile(imgObject.path, imgName)

        if outRgb is not None:
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imgName, outBgr)
