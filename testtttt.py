import os
import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import CascadeClassifier
from cv2 import rectangle
import numpy as np
import pandas as pd
import imutils
from imutils import paths
from skimage.feature import hog, greycomatrix, greycoprops
from skimage import data, segmentation, color, filters, exposure, restoration
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from scipy.signal import convolve2d as conv2
from scipy import ndimage as ndi

from sklearn import metrics
from skimage.data import gravel
from skimage.filters import difference_of_gaussians, window
from scipy.fftpack import fftn, fftshift
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import tensorflow as tf
from zipfile import ZipFile
import pickle

with open('model_random_forest.pickle', 'rb') as f:
    model_rf = pickle.load(f)
    
detector = MTCNN()

filename = 'C:/Users/Farhan/Desktop/Thesis/Software/Splash_Screen/1.jpg'

# fig, axes = plt.subplots(ncols=3, figsize=(15, 3.5))
# ax = axes.ravel()
# ax[0] = plt.subplot(1, 3, 1)
# ax[1] = plt.subplot(1, 3, 2)
# ax[2] = plt.subplot(1, 3, 3)

rawImages  =[]
labels = []

image = cv2.imread(filename)[:,:,::-1]

faces = detector.detect_faces(image)

if list(faces):
    # ax[2] = plt.gca()
    
    twidth, theight = 0, 0
    for face in faces:
        x, y, width, height = face['box']

        if width > twidth and height > theight:
            twidth, theight = width, height
            x2, y2 = x + twidth, y + theight
            # draw a rectangle over the pixels
            roi_image=image[y:y2, x:x2]
            roi_image=roi_image[:,:,::-1]
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
        # rect = Rectangle((x, y), width, height, fill=False, color='yellow')
        # ax[2].add_patch(rect)
#         ax.imshow(image)
#         plt.show()
    selem = disk(20)
    eroded = erosion(roi_gray, selem)
    roi = roi_gray - eroded
    
    size = (150, 150)
    pixels = cv2.resize(roi, size).flatten()
    rawImages.append(pixels)
#     labels.append("happy")
    
    # ax[0].imshow(eroded, cmap=plt.cm.gray)
    # ax[0].set_title('eroded')
    # ax[0].axis('off')

    # ax[1].imshow(roi, cmap=plt.cm.gray)
    # ax[1].set_title('Segmented pic')
    # ax[1].axis('off')
            
#     ax[2].hist(out.ravel(),256,[0,256], color='r')
#     ax[2].set_title('Histogram')
            
    # ax[2].imshow(image)
    # ax[2].axis('off')
    
    # plt.show()
rawImages = np.array(rawImages)
print(rawImages)
