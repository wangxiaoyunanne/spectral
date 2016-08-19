from spectral import *
import glob
import spectral.io.envi as envi
import random
import numpy
from keras.utils import np_utils
files = glob.glob('/home/wangxy/coffee/Coffeebeans_103015/APRIL7_Nugget_103015/S*.bil')
hdrfiles = glob.glob('/home/wangxy/coffee/Coffeebeans_103015/APRIL7_Nugget_103015/S*.hdr')
# open a list to contain all images
imgs_train = []
imgs_test=[]
Y_train = []
Y_test = []
# read images
for i in range(len(files)):
    img = envi.open(hdrfiles[i],files[i])
    #print img.shape
    # cut one hyper spectral graph into 3 
    subimage_1 = img[25:75,177:377]
    subimage_2 = img[25:75,700:900]
    subimage_3 = img[25:75,1233:1433]
    img_set = [subimage_3,subimage_2,subimage_1]
    # get training set and testing set
    test_index = random.randrange(0,3)
    imgs_test.append(img_set[test_index])
    for j in range(3):
        if (j != test_index):
            imgs_train.append(img_set[j])
    Y_train.append(i/2)
    Y_train.append(i/2)
    Y_test.append(i/2)
    #train_index = range(3)- test_index
    #print img_set[test_index].shape

# change output as categorical data    
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
#for i in range(len(files)/2):
                                                              1,1           Top

