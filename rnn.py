#dependencies---------------------------------------------------------------------------------------

import numpy as np #matrix multiplication + maths
import pandas as pd #data management
import os

np.random.seed(1)

#network definitions--------------------------------------------------------------------------------
#X : input matrix of our network composed of sentences with which we want to build a language model.


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

def dtanh(z):
	return 1-np.tanh(z)**2

#RNN
class RNN:
	def __init__ (self,lexicon_size,n_hidden,learning_rate):
		self.lexicon_size=lexicon_size
		self.n_hidden=n_hidden
		self.learning_rate=learning_rate
		self.U= np.random.random((lexicon_size,n_hidden)) #(lexicon_size,n_hidden)
		self.W= np.random.random((n_hidden,n_hidden))#(n_hidden,n_hidden)
		self.V= np.random.random((n_hidden,lexicon_size))#(n_hidden,lexicon_size)
		
	def forward(self,x): #perform prediction of the network regarding an input x, x being the a sequence of word = a sentence
		time_steps=len(x)
		s= np.zeros((time_steps,self.n_hidden))#(time_steps,n_hidden)
		s[-1]=np.zeros(self.n_hidden)#last hidden state
		o= np.zeros((time_steps,self.lexicon_size))#(time_steps,lexicon_size)
		for t in range(0,time_steps):
			s[t]=np.tanh(np.dot(x[t],self.U)+np.dot(self.W,s[t-1])) # X is one hot vector X = (time_steps,lexicon_size) --> s = (time_steps,n_hidden) hidden state is a result of previous hidden state and current input
			o[t]=softmax(np.dot(s[t],V)) # (time_steps,lexicon_size)
		return[s,o]
	def predict(self,x): #pick the value with highest probability at time t (choose the next word)
		s,o=self(forward(x))
		return np.argmax(o,axis=1)

	def total_loss(self,Xtrain,Yreal): #calculate the cross entropy loss. J(Yreal,o). Xtrain being a collection of sentence x
		J=0 #initializse to zero before we start calculation
		#For each sentence
		for i in range(0,len(Yreal)):
			s,o=forward(Xtrain[i])#
			correct_predictions=o[np.arange(len(Yreal[i])),Yreal[i]]
			J=J-np.sum(no.log(correct_predictions))
		return J
	def loss(self,Xtrain,Yreal)
		N=np.sum((len(y_i) for y_i in y))
		return self.total_loss(Xtrain,Yreal)/N

	def bptt(self,X,Yreal): #backpropagation through time using chaining rule for derivative
		time_steps=len(X)
		GJU=np.zeros(self.U.shape)
		GJV=np.zeros(self.V.shape)
		GJW=np.zeros(self.W.shape)
		for t in reversed(np.arrange(time_steps)):#we start we the last time_step
			#calcul of 

			pass

		return [GJU,GJV,GJW]
			

	def training(X,Yreal,learning_rate): #gradient descent for minimization of error
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
		self.wo=np.random.random((1,1))
		self.wc=np.random.random((1,1))
		self.wf=np.random.random((1,1))
		self.wi=np.random.random((1,1))
		self.bf=np.random.random(1)
		self.bi=np.random.random(1)
		self.bo=np.random.random(1)
		self.bc=np.random.random(1)
		#Gradient matrices
		self.go=np.zeros_like(self.wo)
		self.gc=np.zeros_like(self.wc)
		self.gf=np.zeros_like(self.wf)
		self.gi=np.zeros_like(self.wi)


	def forward(self): # compared to RNN, it is just another way to compute the hidden state s[t]
		#forget gate
		f[t]=self.sigmoid(np.dot(self.X,self.wf)+np.dot(self.h_tp,self.wf)+self.bf)
		#input gate
		i[t]=self.sigmoid(np.dot(self.X,self.wi)+np.dot(self.h_tp,self.wi)+self.bi)
		#output gate
		o[t]=self.sigmoid(np.dot(self.X,self.wo)+np.dot(self.h_tp,self.wo)+self.bo)
		#candidate hidden state
		g[t]=self.tanh(np.dot(self.X,self.wc)+np.dot(self.h_tp,wc)+self.bc)
		#internal memory of the unit
		c[t]=np.dot(c[t-1],f[t])+np.dot(g[t],i[t])
		#hidden state
		s[t]=np.dot(tanh(c[t],o[t]))

		return f,i,o,g,c,s

	def gradient(self):
		pass

	def backpropagation(self):
		pass

	def update(self):
		#cell state
		cell_t=f_t*ctp+i_t*c_t
		#hidden state
		h_t=o_t*tanh(c_t)


#GRU
class GRU:
	def __init__(self):
		pass

# if __name__ == '__main__':
# 	print("Training model")
# 	RNN=RNN()
# 	LSTM=LSTM()
# 	print("Model saved")



	