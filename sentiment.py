#sentiment analysis
import numpy as np
import os
import matplotlib.pyplot as plt

os.getcwd()
data=np.load("sentiment.npy")
print(data.shape)
values=data[:,1]
x=[]
y=[]
for i in range(0,len(values)):
	x.append(values[i].score)
	y.append(values[i].magnitude)


#extract data from numpy array


#2d chart of score vs magnitude

plt.scatter(x,y)
plt.show()


