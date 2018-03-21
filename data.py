import numpy as np
import nltk
import pyphen
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import treebank
from collections import Counter
from google.cloud import language
from google.cloud import translate
from google.cloud.language import types
from google.cloud.language import enums
import os
import matplotlib.pyplot as plt
import pandas as pd
import tkinter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/eric/Desktop/smalldatabrains/website/python-speech-reco/key.json"
stopwords=set(nltk.corpus.stopwords.words('french'))
print(stopwords) #implement stopwords for sentiment analysis and words distribution
files=os.listdir("data/")
print(files)
os.chdir("data/")

data=[]
words=[]
size=[]

for file in files:
	f=open(file,'r')
	verses= [line.strip() for line in f]
	data.extend(verses)

print(len(data),"verses in our database")

#tokenize texts
vocabulary_size=2000
unknown_token="unknown_token"
sentence_start_token="sentence_start"
sentence_end_token="sentence_end"

def tokenize(verse):
	words=word_tokenize(verse.lower(),language='french')
	size=len(words)
	return words,size

for line in data:
	liste,word_count=tokenize(line)
	words.extend(liste)
	size.append(word_count)


words=[word for word in words if len(word)>1]
unique=set(words)



print("made of",len(words),"words")
print("with",len(unique),"unique words")



#data analysis & sentiment analysis
analysis=True
if analysis==True:
	#plot word distribution
	fdist=nltk.FreqDist(words)
	fdist.plot(120,title='Most frequent words in French rap lyrics')

	#lenght of words distribution
	length=[]
	for word in words:
		length.append(len(word))

	frequency=Counter(length)
	print(type(frequency))

	labels=[]
	values=[]

	for label,value in frequency.items():
		labels.append(label)
		values.append(value)

	fig1=plt.figure()
	plt.bar(labels,values)
	fig1.suptitle('Distribution of qty of characters per word')
	plt.show()

	#number of word per verse
	word_count_freq=Counter(size)
	for label,value in word_count_freq.items():
		labels.append(label)
		values.append(value)

	fig2=plt.figure()
	plt.bar(labels,values)
	fig2.suptitle('Distribution of qty of words per verse')
	plt.show()

	#number os syllable per verse
	syllable_count=[]
	last_rhyme=[]
	dic=pyphen.Pyphen(lang='fr')
	for verse in data :
		syllables=dic.inserted(verse)
		# print(syllables.split('-')[-1]) study about rhymes
		syllables=syllables.count('-')+1
		syllable_count.append(syllables)
		

	a=[]
	b=[]
	syllable_count_freq=Counter(syllable_count)
	for label,value in syllable_count_freq.items():
		a.append(label)
		b.append(value)
	fig3=plt.figure()
	plt.bar(a,b)
	fig3.suptitle('Distribution of number of syllable per verse')
	plt.show()
	

	#sentiment analysis,(!!! With the use of Google sentiment analysis, becarefull before launching !!!)
	sentiments=[]
	counter=1
	def sentiment(text):
		client=language.LanguageServiceClient()

		document=types.Document(
			content=text,
			type=enums.Document.Type.PLAIN_TEXT)
		sentiment=client.analyze_sentiment(document).document_sentiment
		return sentiment
		
	sentiment_analysis=True
	if sentiment_analysis==True:
		for verse in data:
			try:
				sentiment_returned=sentiment(verse)
				sentiments.append([verse,sentiment_returned])
				counter=counter+1
				if counter %10 == 0:
					print(counter)
					sentiment_array=np.array(sentiments)
					os.chdir(os.pardir)
					np.save("sentiment",sentiments)
					os.chdir("data/")
			except:
				pass


#creating index2word and word2index vectors
word2index=dict((w,i) for i,w in enumerate(unique))
index2char=dict((i,w) for i,w in enumerate(unique))


integer_encoded=[word2index[word] for word in words]

printed=True
if printed==True:
	print(integer_encoded) #integer version (each word correspond to 1 integer value) of the data