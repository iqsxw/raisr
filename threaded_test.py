import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
from gettestargs import gettestargs
from hashkey import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
from multiprocessing import Process
from command import Command
import re

args = gettestargs()

# Define parameters
R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'test'

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

# Read filter from file
filtername = 'filter.p'
if args.filter:
    filtername = args.filter
with open(filtername, "rb") as fp:
    h = pickle.load(fp)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

encoding_opts = {}
encoder = Command('ffmpeg')
if args.input:
    ffprobe = Command('ffprobe')
    ffprobe.append([
        args.input
    ])
    info = ffprobe.execute_then_output()
    lines = info.split('\n')
    for l in lines:
        if 'fps' in l:
            s = l.replace(' ', '').split(':', 1)[1].split(',')
            re.sub(u'(.*?)|[.*?]', '', s[1])
            encoding_opts = {
                'pix_fmt': re.sub(u'\\(.*?\\)', '', s[1]),
                'fps': s[4].replace('fps', ''),
                'resolution': re.sub(u'\\[.*?\\]', '', s[2])
            }
            break

    cmd = Command('ffmpeg')
    cmd.append([
        '-i',
        args.input,
        'test/' + '%d.png'
        ])
    cmd.execute()

    encoder.append([
        '-i',
        'results/%d_result.bmp',
        '-r',
        encoding_opts['fps'],
        '-q',
        '0',
        '-y',
        'test.mp4'
    ])

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))


def UpscaleCoroutine(imageList):
    for index, image in enumerate(imageList):
        print('\r', end='')
        print(' ' * 60, end='')
        print('\rUpscaling image ' + str(index + index) + ' of ' + str(len(imageList)) + ' (' + image + ')')
        origin = cv2.imread(image)
        # Extract only the luminance in YCbCr
        ycrcvorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
        grayorigin = ycrcvorigin[:,:,0]
        # Normalized to [0,1]
        grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
        # Upscale (bilinear interpolation)
        heightLR, widthLR = grayorigin.shape
        heightgridLR = np.linspace(0,heightLR-1,heightLR)
        widthgridLR = np.linspace(0,widthLR-1,widthLR)
        bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, grayorigin, kind='linear')
        heightgridHR = np.linspace(0,heightLR-0.5,heightLR*2)
        widthgridHR = np.linspace(0,widthLR-0.5,widthLR*2)
        upscaledLR = bilinearinterp(widthgridHR, heightgridHR)
        # Calculate predictHR pixels
        heightHR, widthHR = upscaledLR.shape
        predictHR = np.zeros((heightHR-2*margin, widthHR-2*margin))
        operationcount = 0
        totaloperations = (heightHR-2*margin) * (widthHR-2*margin)
        for row in range(margin, heightHR-margin):
            for col in range(margin, widthHR-margin):
                if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                    print('\r|', end='')
                    print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                    print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                    print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                    sys.stdout.flush()
                operationcount += 1
                # Get patch
                patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
                patch = patch.ravel()
                # Get gradient block
                gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
                # Calculate hashkey
                angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
                # Get pixel type
                pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
                predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,pixeltype])
        # Scale back to [0,255]
        predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)
        # Bilinear interpolation on CbCr field
        result = np.zeros((heightHR, widthHR, 3))
        y = ycrcvorigin[:,:,0]
        bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
        result[:,:,0] = bilinearinterp(widthgridHR, heightgridHR)
        cr = ycrcvorigin[:,:,1]
        bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
        result[:,:,1] = bilinearinterp(widthgridHR, heightgridHR)
        cv = ycrcvorigin[:,:,2]
        bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
        result[:,:,2] = bilinearinterp(widthgridHR, heightgridHR)
        result[margin:heightHR-margin,margin:widthHR-margin,0] = predictHR
        result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
        cv2.imwrite('results/' + os.path.splitext(os.path.basename(image))[0] + '_result.bmp', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print('\rCompleted Upscaling image ' + str(index) + ' of ' + str(len(imageList)) + ' (' + image + ')')

imagecount = 1
threads = []
thread_count = 6
dispatches = [[]]
for i in range(0, len(imagelist)):
    if (len(dispatches) < thread_count):
        dispatches.append([imagelist[i]])
    else:
        dispatches[i % thread_count].append(imagelist[i])
    imagecount += 1

for i in range(0, len(dispatches)):
    t = Process(target=UpscaleCoroutine, args=(dispatches[i],))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

encoder.execute()

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
