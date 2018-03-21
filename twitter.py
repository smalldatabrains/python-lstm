import tweepy
import json
import time
from tweepy import Stream
from tweepy.streaming import StreamListener

#Connection

print("Connecting...")
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
print("Connected to tweeter")

#Parameters
msg_limit=100000
language="fr"
party=["politique","insoumis","front de gauche","rassemblement national","front national","fn","en marche","gouvernement","gauche","droite","ps","parti socialiste","les républicains","eelv"]
politics=["mélenchon","jean-marie lepen","marine le pen","marine","macron","poutoux","sarkozy","le pen","hidalgo","hamon","vals"]
hashtag=party+politics

print("Stream started")

class CSVListener(StreamListener):
	def on_data(self,data):
		with open('tweets.txt','a') as tf:
			tf.write(data)
		return True

stream=tweepy.Stream(auth=api.auth,listener=CSVListener())
iterator=stream.filter(track=hashtag,async=True)

#interesting data in the tweets:
#id
#text
#name
#location
#screen_name
#friends_count
#followers_count
#created_at
#utc_offset
#lang
#full_text
#hashtags
