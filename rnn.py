#dependencies---------------------------------------------------------------------------------------

import numpy as np #matrix multiplication + maths
import pandas as pd #data management
import os

#network definitions--------------------------------------------------------------------------------
# Z : output vector for each layer
# X : input vector of size vocabulary_size
# W : Weights matrix
# a : values vector after applying the sigmoid function to Z, a becomes the new input vector for the next layer
# Yreal : real value vector of the ouput
# Yth : theoritical value vector calculated by our network after forward propagation
# J : costfunction, return single value
# gradient : gradient of J regarding the Weights
# RNN cell:
# LSTM cell : 
	#xt
	#ct
	#ctp
	#ft
	#it
	#ot
	#cell_t
# forget gate :
# input gate :
# output gate :
# learning rate lr: decaying learning rate (for gradient descent) with RMSprop method


#usual functions
def normalize(X):
	return X/np.amax(X)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def softmax(z):
	softmax = np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)),axis=1,keepdims=True)
	return softmax
		
def dsigmoid(z):
	return np.exp(-z)/(1+np.exp(-z))**2

def tanh(z):
	return np.tanh(z)

def dtanh(z):
	return 1-np.tanh(z)**2

#data preparation functions
def tokenize():
	pass

def word2vec():
	pass

def vec2word():
	pass

#RNN
class RNN:
	def __init__ (self,vocabulary_size,n_hidden,learning_rate):
		self.vocabulary_size=vocabulary_size
		self.n_hidden=n_hidden
		self.learning_rate=learning_rate
		self.U= np.random.random((vocabulary_size,n_hidden)) #(vocabulary_size,n_hidden)
		self.W= np.random.random((n_hidden,n_hidden))#(n_hidden,n_hidden)
		self.V= np.random.random((n_hidden,vocabulary_size))#(n_hidden,vocabulary_size)
		self.s_t=
		self.s_tp=
		self.x_t=
	def forward(self,X):
		time_steps=len(X)
		s_t=tanh(np.dot(x_t,U)+np.dot(W,s_tp))
		o_t=softmax(np.dot(s_t,V))
		return s_t,o_t
	def loss(self,X,Yreal):
		pass

#LSTM

class LSTM:
	def __init__ (self,X,Yreal,recurrence,learning_rate):
		#input and output
		self.X=np.zeros(X.shape[0]) #input series
		self.Yreal=np.zeros(Yreal.shape[0]) #next data coming right after the serie
		#learning rate
		self.learning_rate=learning_rate
		#Weights matrices and biases
		self.wo=np.random.random(())
		self.wc=np.random.random(())
		self.wf=np.random.random(())
		self.wi=np.random.random(())
		self.bf=np.random.random()
		self.bi=np.random.random()
		self.bo=np.random.random()
		self.bc=np.random.random()
		#Gradient matrices
		self.go=np.zeros_like(self.wo)
		self.gc=np.zeros_like(self.wc)
		self.gf=np.zeros_like(self.wf)
		self.gi=np.zeros_like(self.wi)


	def forward(self):
		#forget gate
		f_t=self.sigmoid(np.dot(self.X,self.wf)+np.dot(self.h_tp,self.wf)+self.bf)
		#input gate
		i_t=self.sigmoid(np.dot(self.X,self.wi)+np.dot(self.h_tp,self.wi)+self.bi)
		#output gate
		o_t=self.sigmoid(np.dot(self.X,self.wo)+np.dot(self.h_tp,self.wo)+self.bo)
		#cell state
		c_t=self.tanh(np.dot(self.X,self.wc)+np.dot(self.h_tp,wc)+self.bc)

		return ft,it,ot,cell_t

	def gradient(self):
		pass

	def backpropagation(self):
		pass

	def update(self):
		#cell state
		cell_t=f_t*ctp+i_t*c_t
		#hidden state
		h_t=o_t*tanh(c_t)



if __name__ == '__main__':
	print("Training model")
	RNN=RNN()
	LSTM=LSTM()
	print("Model saved")



	