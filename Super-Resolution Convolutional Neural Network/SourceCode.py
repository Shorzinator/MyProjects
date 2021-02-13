import sys
import keras
import cv2
import numpy as np
import matplotlib
import skimage
import math
import os

print('Python {}'.format(sys.version))
print('Keras {}'.format(keras.__version__))
print('OpenCV {}'.format(cv2.__version__))
print('NumPy {}'.format(np.__version__))
print('MatPlotLib {}'.format(matplotlib.__version__))
print('Skimage {}'.format(skimage.__version__))

from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt


#define a function for peak signal to noise ratio
def psnr(target,ref):

    #assume RGB/BGR image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')                                                    #'C' is to flatten in row major, 'F' is to flatten by column major

    rmse = math.sqrt(np.mean(diff ** 2.0))

    return 20 * math.log10(255.0/rmse)

#define function for the mean squared error (MSE)
def mse(target,ref):
    err = np.sum((target.astype('float')-ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])                             #total number of pixels that we have

    # MSE is the sum of the squared difference between the two images

    return err

#define a function that evaluates all three  image quality metrics
def compare_images(target,ref):
    scores = []
    scores.append(psnr(target,ref))
    scores.append(mse(target,ref))
    scores.append(ssim(target,ref,multichannel = True))
    # or scores = [psnr(target, ref), mse(target, ref), ssim(target, ref, multichannel=True)]

    return scores


#prepare degraded images by introducting quality distortion by resizing images
def prepare_images(factor):
    path = './SRCNN/source'
    #loop through the file in the directory
    for files in os.listdir(path):
        #open the file
        img = cv2.imread(path + '/' + files)

        #find old and new image dimaensions
        h, w, c = img.shape
        new_height = h /factor
        new_width = w / factor

        #resize the image - down
        img = (cv2.resize(img, (int(new_width), int(new_height)), interpolation=cv2.INTER_LINEAR))

        #resize the image - up
        img = (cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR))

        # save the image

        print('Saving {}'.format(files))

        cv2.imwrite('./SRCNN/images/{}'.format(files), img)

prepare_images(2)

#testing the generated images using quality metrics

for file in os.listdir('./SRCNN/images/'):
    target = cv2.imread('./SRCNN/images/{}'.format(file))
    ref = cv2.imread('./SRCNN/source/{}'.format(file))
    scores = compare_images(target, ref)
    print('{}\n PSNR: {}\n MSE: {}\n SSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))


#define the SRCNN model

def model():

    # define model type
    SRCNN = Sequential()

    #add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (9,9), kernel_initializer='glorot_uniform',
             activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64,  kernel_size = (3,3), kernel_initializer='glorot_uniform',
             activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1,   kernel_size = (5,5), kernel_initializer='glorot_uniform',
             activation='linear', padding='valid', use_bias=True))

    #define optimizer
    adam = Adam(lr=0.0001)

    #compile mode
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN


#pre-processing functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale) #  what this is ensuring that our image is going to be divisible by scale.
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]  # to even out the borders and make all the images of the same size
    return img


# define main prediction function

def predict(image_path):

    # load the srcnn model with weights
    srcnn = model()
    srcnn.load_weights('./SRCNN/3051crop_weight_200.h5')

    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('./SRCNN/source/{}'.format(file))

    # preprocess the image with modcrop
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)

    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)

    # create image slice and normalize
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    CNN
    # perform super-resolution with srcnn
    pre = srcnn.predict(Y, batch_size=1)

    # post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)

    # copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    # remove border from reference and degraged image
    ref = shave(ref.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)

    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref))
    scores.append(compare_images(output, ref))

    # return images and scores
    return ref, degraded, output, scores


for file in os.listdir('./SRCNN/images/'):
   #perform super-resolution
    ref, degraded, output, scores = predict('./SRCNN/images/{}'.format(file))


    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[1].set_xlabel('PSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[0][0], scores[0][1], scores[0][2]))
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    axs[2].set_xlabel('PSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[1][0], scores[1][1], scores[1][2]))

    print('Saving {}'.format(file))
    fig.savefig('./SRCNN/output/{}.png'.format(os.path.splitext(file)[0]))
