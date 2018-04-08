
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from skimage import io, color,transform



im=Image.open('C:\\Anaconda3\\pictures\\123.jpg')
plt.figure("dog")
plt.imshow(im)
plt.show()
print (im.mode)
'''convert to array'''
im1=np.array(im)
print (im1.shape)
im1=im1.reshape(im1.shape+(1,))
print(im1.shape)
im2=im1[1][:,:,0]
print(im2.shape)
print (len(im1))
'''resize the image'''
# ~ im1=transform.resize(im1,(299,299,3),mode='constant')
# ~ print (im1.shape)
# ~ plt.imshow(im1)
# ~ plt.show()
'''convert to lab'''
# ~ lab = color.rgb2lab(1.0/255*im1)
# ~ print(lab.shape[:,:,0])
# ~ #lab=np.expand_dims(lab, axis=0)#add dimantion as axis
# ~ print(lab[:,:,0].shape+(1,))#add a dimantion in the end 
# ~ print (lab.shape)
# ~ plt.imshow(lab[:,:,0],'binary')
# ~ plt.show()
# ~ print (lab[:,:,0])
'''back to rgb'''
# ~ lab=color.lab2rgb(lab)
# ~ plt.imshow(lab)
# ~ plt.show()

# ~ img=np.zeros(im.shape)
'''change channels'''
# ~ img[:,:,0]=im[:,:,2]change channels
# ~ img[:,:,1]=im[:,:,0]
# ~ img[:,:,2]=im[:,:,1]
# ~ plt.imshow(img)
# ~ plt.show()
#fig = plt.figure()  
#plotwindow = fig.add_subplot(111)  
#plt.imshow(X_train[1,0] , cmap = 'binary')  
#plt.show()
#print (y_train.shape)
#print (y_train[10])
