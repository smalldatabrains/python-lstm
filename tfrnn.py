#a tensorflow implementation of Rnn and Lstm networks

#dependencies
import tensorflow as tf
import numpy as np


#placeholders


#initialization
hidden_state=
current_state=
state=
probabilities
loss=0


#training

for current_batch_of_words in words_in_dataset:
	numpy_state,current_loss=session.run([final_state,loss],
		feed_dict={initial_state:numpy_state,words:current_batch_of_words})
	total_loss +=current_loss