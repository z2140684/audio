import numpy as np
import os
import re
from pylab import*
from scipy.io import wavfile
import numpy as np
from math import floor
from keras.utils import np_utils
from keras.layers import Input,Dense,Dropout,Activation,Flatten,Reshape,MaxPooling2D
from keras.layers import Convolution1D,BatchNormalization,LSTM,Convolution2D
from keras.models import load_model,Model
import h5py
from keras.callbacks import TensorBoard,ModelCheckpoint,Callback
import matplotlib.pyplot as plt

def get_path_label_name():
	X_Pall=[]
	y_all=[]

	
	for directory in os.listdir("D:/code/MINI_DATA/speech_command/class/"):
		for filename in os.listdir("D:/code/MINI_DATA/speech_command/class/"+directory):
				path="D:/code/MINI_DATA/speech_command/class/"+directory+"/"+filename
				X_Pall.append(path)
				y_all.append(directory)

	return np.array(X_Pall),np.array(y_all)#,np.array(X_Pvalid),np.array(y_valid),np.array(X_Ptest),np.array(y_test)

def get_indicate1(y):
	'''class to number'''
	indicates={}
	i=0
	for m in set(y):
		indicates[m]=i
		i+=1

	return indicates
def get_indicate2(y):
	'''number to class'''
	indicates={}
	i=0
	for m in set(y):
		indicates[i]=m
		i+=1
	return indicates
	
	
def transfer_y(y,indicates):
	for i in range(len(y)):
		y[i]=indicates[y[i]]
		
	return y
	
def FFT(path):
	'''windo is 80 sampFreq'''
	sampFreq, snd = wavfile.read(path)
	length=snd.shape[0]
	snd = snd / (2.**15)
	steps=[]
	for i in range(0,floor(length/80)):
		s1=snd[i*80:(i+1)*80]
		n = len(s1)
		p = fft(s1) 
		nUniquePts = ceil((n+1)/2.0)
		p = p[0:int(nUniquePts)]
		p = abs(p)
		p = p / float(n)    #除以采样点数，去除幅度对信号长度或采样频率的依赖
		           #求平方得到能量
		#奇nfft排除奈奎斯特点
		if n % 2 > 0: #fft点数为奇
			p[1:len(p)] = p[1:len(p)]*2
		else: #fft点数为偶
			p[1:len(p)-1] = p[1:len(p)-1] * 2
		
		steps.append(p)
	steps=np.array(steps)
	if int(steps.shape[0])<200:
		fix=np.zeros((200-int(steps.shape[0]),41))
		final=np.vstack((steps,fix))
		return np.array(final)
	elif int(steps.shape[0])==200:
		return np.array(steps)
	else:
		print ("lost a big one")
		


def get_Xdata(X_Pwavset):
	X_train=np.zeros((X_Pwavset.shape[0],200,41))
	i=0
	for wavP in X_Pwavset:
		X_train[i,:]=FFT(wavP)
		i+=1
	return np.array(X_train)
	
def shuffle_data(X,y):
	index=np.arange(len(X))
	np.random.shuffle(index)
	X=X[index]
	y=y[index]
	return X,y
	
	
def save_fft_data(X_Pall,Y_all):
	print('use FFT to compute')
	maxnum=X_Pall.shape[0]
	itera_train=floor(0.8*maxnum/20)
	itera_valid=floor(0.1*maxnum/20)
	itera_test=floor(0.1*maxnum/20)
	for i in range(itera_train):
		X_train=get_Xdata(X_Pall[i*20:(i+1)*20])
		print('train data '+str(i)+' is saved as '+str(X_train.shape))
		np.save("D:/code/MINI_DATA/speech_command/X_train"+str(i)+".npy",X_train)
		np.save("D:/code/MINI_DATA/speech_command/Y_train"+str(i)+".npy",Y_all[i*20:(i+1)*20])
	j=0	
	for i in range(itera_train,itera_train+itera_valid):
		X_valid=get_Xdata(X_Pall[i*20:(i+1)*20])
		print('valid data '+str(j)+' is saved as '+str(X_valid.shape))
		np.save("D:/code/MINI_DATA/speech_command/X_valid"+str(j)+".npy",X_valid)
		np.save("D:/code/MINI_DATA/speech_command/Y_valid"+str(j)+".npy",Y_all[i*20:(i+1)*20])
		j+=1
	j=0
	for i in range((itera_train+itera_valid),floor(maxnum/20)):
		X_test=get_Xdata(X_Pall[i*20:(i+1)*20])
		print('test data '+str(j)+' is saved as '+str(X_test.shape))
		np.save("D:/code/MINI_DATA/speech_command/X_test"+str(j)+".npy",X_test)
		np.save("D:/code/MINI_DATA/speech_command/Y_test"+str(j)+".npy",Y_all[i*20:(i+1)*20])
		j+=1
	
def new_data():
	X_Pall,y_all=get_path_label_name()#,X_Pvalid,y_valid,X_Ptest,y_test
	
	
	class_2num=get_indicate1(y_all)
	num_2class=get_indicate2(y_all)
	print(class_2num)
	Y_all=transfer_y(y_all,class_2num)
	
	X_Pall,Y_all=shuffle_data(X_Pall,Y_all)

	Y_all=np_utils.to_categorical(Y_all,30)
	
	return (X_Pall,Y_all)

def Tdata_generate():
	i=0
	while(1):
		path_x="D:/code/MINI_DATA/speech_command/X_train"+str(i)+".npy"
		path_y="D:/code/MINI_DATA/speech_command/Y_train"+str(i)+".npy"
		X_train=np.load(path_x)
		Y_train=np.load(path_y)
		#X_train=(X_train-X_train.mean())/X_train.max()
		i+=1
		i=i%2588
		yield(X_train,Y_train)
			
			
def Vdata_generate():
	i=0
	while(1):
		path_x="D:/code/MINI_DATA/speech_command/X_valid"+str(i)+".npy"
		path_y="D:/code/MINI_DATA/speech_command/Y_valid"+str(i)+".npy"
		X_valid=np.load(path_x)
		Y_valid=np.load(path_y)
		#X_valid=(X_valid-X_valid.mean())/X_valid.max()
		i+=1
		i=i%323
		yield(X_valid,Y_valid)
		
def model_architecture(input_shape=(200,41)):
	X_input=Input(input_shape)
	X=Convolution1D(192,4,strides=2)(X_input)
	X=BatchNormalization()(X)
	X=Activation('relu')(X)
	X=Dropout(0.5)(X)
	X=LSTM(128,return_sequences=True)(X)
	X=BatchNormalization()(X)
	X=Dropout(0.5)(X)
	X=LSTM(128,return_sequences=False)(X)#dropout_W=0.2,dropout_U=0.2
	X=Dropout(0.5)(X)
	Xout=Dense(30,activation='softmax')(X)
	model=Model(inputs=X_input,outputs=Xout)
	return model
	
def model_architecture2(input_shape=(200,41)):
	X_input=Input(input_shape)
	X=Reshape([200,41,1])(X_input)
	X=Convolution2D(32,(3,3),padding='same')(X)
	X=BatchNormalization()(X)
	X=Activation('relu')(X)
	X=MaxPooling2D(pool_size=(2,1))(X)#100,41
	X=Convolution2D(32,(7,2))(X)#94,40
	X=BatchNormalization()(X)
	X=Activation('relu')(X)
	X=MaxPooling2D(pool_size=(2,1))(X)#47,20
	X=Convolution2D(16,(6,3))(X)#42,18
	X=BatchNormalization()(X)
	X=Activation('relu')(X)
	X=Flatten()(X)
	X=Dense(64,activation='relu')(X)
	X=Dropout(0.5)(X)
	X_out=Dense(30,activation='softmax')(X)
	model=Model(inputs=X_input,outputs=X_out)
	return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def new_train():
	model=model_architecture()
	check=ModelCheckpoint(filepath='D:/code/MINI_DATA/speech_command/model/speech.h5')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit_generator(Tdata_generate(),epochs=1,steps_per_epoch=2588,callbacks=[check])
	model.save('D:/code/MINI_DATA/speech_command/model/speech.h5')
	

def load_train():
	model=load_model('D:/code/MINI_DATA/speech_command/model/speech.h5')
	check=ModelCheckpoint(filepath='D:/code/MINI_DATA/speech_command/model/speech.h5')
	history = LossHistory()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.fit_generator(Vdata_generate(),epochs=20,steps_per_epoch=646,callbacks=[check])
	model.fit_generator(Tdata_generate(),epochs=3,steps_per_epoch=2588,callbacks=[check,history],validation_data=Vdata_generate(),validation_steps=324)
	model.save('D:/code/MINI_DATA/speech_command/model/speech.h5')
	history.loss_plot('epoch')
	
	
def evalution():
	model=load_model('D:/code/MINI_DATA/speech_command/model/speech.h5')
	for i in range(0,324):
		path_x="D:/code/MINI_DATA/speech_command/X_test"+str(i)+".npy"
		path_y="D:/code/MINI_DATA/speech_command/Y_test"+str(i)+".npy"
		X_test=np.load(path_x)
		Y_test=np.load(path_y)
		X_test=(X_test-X_test.mean())/X_test.max()
		score=model.evaluate(X_test,Y_test,verbose=0)
		print("evaluation is:"+str(score))
	
'''convet audio to fft and save as npy files'''	
# ~ X_Pall,Y_all=new_data()

# ~ save_fft_data(X_Pall,Y_all)


'''load npy files and train '''
#new_train()
load_train()
# ~ #evalution()



# ~ X_train=np.load("D:/code/MINI_DATA/speech_command/X_train0.npy")
# ~ X_valid=np.load("D:/code/MINI_DATA/speech_command/X_valid0.npy")
# ~ X_test=np.load("D:/code/MINI_DATA/speech_command/X_test0.npy")
# ~ Y_train=np.load("D:/code/MINI_DATA/speech_command/Y_train0.npy")
# ~ Y_valid=np.load("D:/code/MINI_DATA/speech_command/Y_valid0.npy")
# ~ Y_test=np.load("D:/code/MINI_DATA/speech_command/Y_test0.npy")			
# ~ print(X_Ptrain.shape)
# ~ print(Y_train.shape)
# ~ print(X_Pvalid.shape)
# ~ print(Y_valid.shape)
# ~ print(X_Ptest.shape)
# ~ print(Y_test.shape)


