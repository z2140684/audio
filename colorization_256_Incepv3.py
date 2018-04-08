from keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import tensorflow as tf
from keras.layers import Input,Dense,Dropout,Activation,Flatten,RepeatVector,concatenate
from keras.layers import Convolution2D,UpSampling2D,Reshape,BatchNormalization,Add
#from keras.applications import inception_v3
#from keras.applications import InceptionV3
#from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from PIL import Image
from skimage import color,transform,io
from keras.models import load_model,Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard,ModelCheckpoint
import h5py
#from keras.utils import multi_gpu_model
from keras.applications import imagenet_utils


classnet=InceptionV3(weights='imagenet', include_top=False)#imagenet
classnet.graph=tf.get_default_graph()#computation graph belonged to classnet


def model_architecture(image_shape=(256,256,1,),class_shape=(8,8,2048)):
	class_input=Input(class_shape)
	
	'''低级特征'''
	low_input=Input(image_shape)
	low=Convolution2D(64,(3,3),strides=(2,2),padding='same')(low_input)
	low=BatchNormalization()(low)
	low=Activation('relu')(low)
	
	low=Convolution2D(128,(3,3),padding='same')(low)#128*128*128
	low=BatchNormalization()(low)
	lowf=Activation('relu')(low)
	
	low=Convolution2D(128,(3,3),strides=(2,2),padding='same')(lowf)
	low=BatchNormalization()(low)
	low=Activation('relu')(low)
	
	low=Convolution2D(256,(3,3),padding='same')(low)
	low=BatchNormalization()(low)
	low=Activation('relu')(low)
	
	low=Convolution2D(256,(3,3),strides=(2,2),padding='same')(low)
	low=BatchNormalization()(low)
	low=Activation('relu')(low)
	
	low=Convolution2D(512,(3,3),padding='same')(low)
	low=BatchNormalization()(low)
	low_out=Activation('relu')(low)
	
	'''中级特征'''
	mid=Convolution2D(512,(3,3),padding='same')(low_out)
	mid=BatchNormalization()(mid)
	mid=Activation('relu')(mid)
	
	mid=Convolution2D(256,(3,3),padding='same')(mid)#32,32,256
	mid=BatchNormalization()(mid)
	mid_out=Activation('relu')(mid)
	
	'''inceptionV3特征'''
	clas=Convolution2D(512,(1,1),padding='same')(class_input) #8*8*512
	clas=Flatten()(clas)
	clas=Dense(1024)(clas)
	clas=BatchNormalization()(clas)
	clas=Activation('relu')(clas)
	clas=Dense(512)(clas)
	clas=BatchNormalization()(clas)
	clas=Activation('relu')(clas)
	clas=Dense(256)(clas)
	clas=BatchNormalization()(clas)
	clas=Activation('relu')(clas)
	clas=RepeatVector(32*32)(clas)
	clas_out=Reshape(([32,32,256]))(clas)
	
	'''融合层'''
	fusion=concatenate([mid_out,clas_out],axis=3)#32,32,256*2
	fusion=Convolution2D(256,(1,1),padding='same')(fusion)#32,32,256
	fusion=BatchNormalization()(fusion)
	fusion_out=Activation('relu')(fusion)
	
	'''上色层'''
	color=Convolution2D(128,(3,3),padding='same')(fusion_out)
	color=BatchNormalization()(color)
	color=Activation('relu')(color)
	
	color=UpSampling2D((2,2))(color)
	
	color=Convolution2D(64,(3,3),padding='same')(color)
	color=BatchNormalization()(color)
	color=Activation('relu')(color)
	
	color=Convolution2D(64,(3,3),padding='same')(color)
	color=BatchNormalization()(color)
	color=Activation('relu')(color)
	
	colorf=UpSampling2D((2,2))(color)#128*128*64
	
					'''底层特征融合'''
	LHfusion=Convolution2D(64,(1,1))(lowf)#128*128*64
	LHfusion=Add()([LHfusion,colorf])
	LHfusion=Convolution2D(32,(3,3),padding='same')(LHfusion)
	color=BatchNormalization()(color)
	color=Activation('relu')(color)
	
	color=Convolution2D(32,(3,3),padding='same')(LHfusion)#128*128*32
	color=BatchNormalization()(color)
	color=Activation('relu')(color)
	
	color=Convolution2D(2,(3,3),padding='same',activation='tanh')(color)
	
	color_out=UpSampling2D((2,2))(color)
	
	model=Model(inputs=[low_input,class_input],outputs=color_out)#build a computation graph named model
	return model


def classification_result(gray_image):
	'''get InceptionV3's prediction'''
	gray_image_set=[] 
	for m in gray_image:
		m=transform.resize(m,(299,299,3),mode='constant')
		gray_image_set.append(m)
	gray_image_set=np.array(gray_image_set)
	gray_image_set=imagenet_utils.preprocess_input(gray_image_set)
	with classnet.graph.as_default():
		classification=classnet.predict(gray_image_set)
	return classification


def batch_image_generate(directory='F:/Data/test/testSet_resize',batch_size=40):#40,64
	image_data=image.ImageDataGenerator(shear_range=0.4,
        horizontal_flip=True)#
	for batch in image_data.flow_from_directory(directory,batch_size=batch_size):
		batch=np.array(batch[0],dtype=float)#？？
		batch=1.0/255*batch
		gray_image=color.gray2rgb(color.rgb2gray(batch))
		classification=classification_result(gray_image)
		lab_image=color.rgb2lab(batch)
		x_train=lab_image[:,:,:,0]
		x_train=x_train.reshape(x_train.shape+(1,))
		y_train=lab_image[:,:,:,1:]/128
		yield ([x_train,classification],y_train)
		
def color_image(path_in='F:/Data/test/123/',path_out='F:/Data/test/try_out/',path_model='D:/code/model/try1.h5'):
	image_collecter=[]
	for image_name in os.listdir(path_in):
		image=Image.open(path_in+image_name)
		image=np.array(image,dtype=float)
		image_collecter.append(image)
	image_collecter=np.array(image_collecter)
	image_collecter=1.0/255*image_collecter
	image_low_input=color.rgb2lab(image_collecter)[:,:,:,0]
	image_low_input=image_low_input.reshape(image_low_input.shape+(1,))
	image_class_input=color.gray2rgb(color.rgb2gray(image_collecter))
	classification=classification_result(image_class_input)		
	model=load_model(path_model)
	output=model.predict([image_low_input,classification])#two input have to be in []
	output=output*128
	current_image=np.zeros((256,256,3))
	for i in range(len(output)):
		 current_image[:,:,1:]=output[i]
		 current_image[:,:,0]=image_low_input[i,:,:,0]
		 current_image=color.lab2rgb(current_image)
		 io.imsave(path_out+'2'+str(i)+'.png',current_image)


def new_train(directory='D:/123'):
	model=model_architecture()
	tensorboard =TensorBoard(log_dir="D:/code/model/tensor_graph_Incep3")#--logdir=D://code//model//tensor_graph
	model.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
	model.fit_generator(batch_image_generate(directory),callbacks=[tensorboard],epochs=1,steps_per_epoch=8212)
	model.save('D:/code/model/color_Ince3.h5')	
	print ('good job boy')	
	#plot_model(model, to_file='D:/code/model/try_image.png')
	
	
def load_train(path='D:/code/model/color_Ince3.h5',directory='D:/123'):
	model=load_model(path)
	print('model loaded')
	tensorboard =TensorBoard(log_dir="D:/code/model/tensor_graph_Incep3")
	check=ModelCheckpoint(filepath='D:/code/model/color_Ince3.h5')
	model.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
	model.fit_generator(batch_image_generate(directory),callbacks=[tensorboard,check],epochs=6,steps_per_epoch=8212)
	model.save('D:/code/model/color_Ince3.h5')
	print('model saved')
	print ('good job boy')
	#plot_model(model, to_file='D:/code/model/try_image.png')

def multiply_new_train(directory='F:/Data/test/try'):
	model=model_architecture()
	parallel_model = multi_gpu_model(model, gpus=8)
	parallel_model.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
	parallel_model.fit_generator(batch_image_generate(directory),epochs=1,steps_per_epoch=15)
	model.save('D:/code/model/try.h5')		
	#plot_model(model, to_file='D:/code/model/try_image.png')
	
def load_multiply_train(path='D:/code/model/try.h5',directory='F:/Data/test/try'):
	model=load_model(path)
	parallel_model = multi_gpu_model(model, gpus=8)
	parallel_model.compile(optimizer='adadelta',loss='mse')
	parallel_model.fit_generator(batch_image_generate(directory),epochs=2,steps_per_epoch=20)
	model.save('D:/code/model/try.h5')		
	#plot_model(model, to_file='D:/code/model/try_image.png')
	
new_train()#graphviz
#load_train()
#color_image()

