import numpy as np
from keras.preprocessing import sequence,text
from keras.models import load_model,Model
from keras.layers import Dense, Activation, Embedding,Dropout
from keras.layers import LSTM,Input
from keras.datasets import imdb
import os
from numpy import loadtxt
def raw_data():
	X=[]
	for filename in os.listdir('D:/code/MINI_DATA/IMDB/aclImdb/train/neg/'):#read and get all filenames in diract
		with open('D:/code/MINI_DATA/IMDB/aclImdb/train/neg/'+filename,encoding='utf-8') as f:#use utf-8 to decode text file
			content=f.read()
			words=text.text_to_word_sequence(content)#keras tool to preprosses sentence and convert it to singal word in list
			X.append(words)
	print(len(X))
	for filename in os.listdir('D:/code/MINI_DATA/IMDB/aclImdb/train/pos/'):
		with open('D:/code/MINI_DATA/IMDB/aclImdb/train/pos/'+filename,encoding='utf-8') as f:
			content=f.read()
			words=text.text_to_word_sequence(content)
			X.append(words)
	print(len(X))
	
	word_index = imdb.get_word_index()#get a dictionary of word_indicates
	for i in range(len(X)):
		for j in range(len(X[i])):
			X[i][j]=word_index[X[i][j]] if X[i][j] in word_index else 0#get each indicate
			
	X=np.array(X)#now X is half positive examples and half neg examples
	y=np.zeros((25000,1))
	y[12500:,:]=1#set late half as 1 to match pos examples
	indices=np.arange(len(X))
	np.random.shuffle(indices)#shuffle the examples
	X=X[indices]
	y=y[indices]
	print(X.shape)
	print(y.shape)
	return X,y

def raw_test():
	X=[]
	for filename in os.listdir('D:/code/MINI_DATA/IMDB/aclImdb/test/neg/'):
		with open('D:/code/MINI_DATA/IMDB/aclImdb/test/neg/'+filename,encoding='utf-8') as f:
			content=f.read()
			words=text.text_to_word_sequence(content)
			X.append(words)
	print(len(X))
	for filename in os.listdir('D:/code/MINI_DATA/IMDB/aclImdb/test/pos/'):
		with open('D:/code/MINI_DATA/IMDB/aclImdb/test/pos/'+filename,encoding='utf-8') as f:
			content=f.read()
			words=text.text_to_word_sequence(content)
			X.append(words)
	print(len(X))
	
	word_index = imdb.get_word_index()
	all_ind=[]
	for i in range(len(X)):
		ind=[]
		for j in range(len(X[i])):
			X[i][j]=word_index[X[i][j]] if X[i][j] in word_index else 0
			
	X=np.array(X)
	y=np.zeros((25000,1))
	y[12500:,:]=1
	indices=np.arange(len(X))
	np.random.shuffle(indices)
	X=X[indices]
	y=y[indices]
	print(X.shape)
	print(y.shape)
	return X,y
	
def get_data():
	(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)#words rankde by appearing frequence 
	y_train=y_train.reshape(25000,1)
	y_test=y_test.reshape(25000,1)
	print (X_train.shape)
	print (y_train.shape)
	print (X_test.shape)
	print (X_test.shape)
	print (X_train[1])
	return X_train, y_train, X_test, y_test
	
	
def reshape(X_train,X_test):
	X_train=sequence.pad_sequences(X_train,maxlen=80)#padding each exaple as same length
	X_test=sequence.pad_sequences(X_test,maxlen=80)
	print (X_train.shape)
	print (X_test.shape)
	print (X_train[1])
	return X_train,X_test
    
    
def get_model(input_shape=((80,))):#should set (maxlen,) as input_shape
	input_x=Input(input_shape)
	X=Embedding(25000,128)(input_x)
	X=LSTM(100,dropout_W=0.2,dropout_U=0.2)(X)#100 is the number of W(hinden unit),can change to any number 
	X=Dropout(0.5)(X)
	X=Dense(1)(X)
	Xout=Activation('sigmoid')(X)
	model=Model(inputs=input_x,outputs=Xout)
	return model

def new_train(X_train,y_train,X_test,y_test):
	model=get_model()
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, batch_size=32, nb_epoch=1,validation_data=(X_test, y_test))        
	model.save('D:/code/MINI_DATA/IMDB/imdb_model.h5')  
    
def load_train(X_train,y_train,X_test,y_test):
	model=load_model('D:/code/MINI_DATA/IMDB/imdb_model.h5')
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, batch_size=32, nb_epoch=10)     
	model.save('D:/code/MINI_DATA/IMDB/imdb_model.h5')  	

def evaluation(X_test,y_test):
	model=load_model('D:/code/MINI_DATA/IMDB/imdb_model.h5')
	print('begin')
	acc=model.evaluate(X_test,y_test,verbose=0) 
	print ('evaluation: '+str(acc))
	
#X_train,y_train,X_test,y_test=get_data()
#X_train,X_test=reshape(X_train,X_test)
#load_train(X_train,y_train,X_test,y_test)
#evaluation(X_test,y_test)
X_train,y_train=raw_data()
X_test,y_test=raw_test()
X_train,X_test=reshape(X_train,X_test)
print(X_train.shape)
#new_train(X_train,y_train,X_test,y_test)
#print(evaluation(X_test,y_test))
#load_train(X_train,y_train,X_test,y_test)
#get_data()
