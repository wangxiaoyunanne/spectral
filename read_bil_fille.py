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
# change output as categorical data    
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
X_train= numpy.zeros(0)
for i in range(0,60):
    print i
   # length = imgs_train[i].shape[0]
   # width = imgs_train[i].shape[1]
   # dim = imgs_train[i].shape[2]
    # reshape img, HORIZONTALLY strench the img, without changing the spectral dim.
   # reshaped_img = numpy.asarray(imgs_train[i].reshape(length*width, dim), 
   #                              dtype=theano.config.floatX)
   # pca = PCA(n_components=n_principle)
   # reshaped_img   
   # pca_img = pca.fit_transform(reshaped_img)
    PCA_img,_,_ = PCA_tramsform_img(imgs_train[i])
    X_train = numpy.append(X_train,PCA_img)

X_test = numpy.zeros(0)
for i in range (0,30):
    print i
    PCA_img,_,_ = PCA_tramsform_img(imgs_test[i])
    X_test = numpy.append(X_test,PCA_img)




seed = 7
numpy.random.seed(seed)


X_train = X_train.reshape(60,3,50,60)
X_test = X_test.reshape(30,3,50,60)

def larger_model():
    model = Sequential()
    model.add(Convolution2D(60,8,8,border_mode  ='valid',input_shape = (3,50,60)  ,activation='relu'))
    model.add(MaxPooling2D(pool_size = (4,4)))
    model.add(Convolution2D(30,5,5,border_mode  ='valid',input_shape = (3,50,60)  ,activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.04))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense (50,activation = 'relu'))
    model.add(Dense(15,activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])
    return model


model = larger_model()
model.fit(X_train,Y_train,validation_data = (X_test,Y_test), nb_epoch = 10, batch_size = 200, verbose = 2)
scores =  model.evaluate(X_test,Y_test,verbose = 0)
print ('err %.2f%%' % (100-scores[1]*100))
                                              
                                                              1,1           Top

