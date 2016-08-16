from spectral import *
import glob
import spectral.io.envi as envi

files = glob.glob('/home/wangxy/coffee/Coffeebeans_103015/APRIL7_Nugget_103015/*.bil')
hdrfiles = glob.glob('/home/wangxy/coffee/Coffeebeans_103015/APRIL7_Nugget_103015/*.hdr')
# open a list to contain all images
imgs = []
# read images
for i in range(len(files)):
    img = envi.open(hdrfiles[i],files[i])
    print img.shape
    imgs.append(img)

