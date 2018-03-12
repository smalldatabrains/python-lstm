import tweepy
import json
import time
from tweepy import Stream
from tweepy.streaming import StreamListener

#Connection
consumer_key="4JwQALQi9wOOaEwUEYKZcsWE8"
consumer_secret="wJtMKtG0meyX3d0vWNkNXidDmuA1pLHAh6GK4niZ56z17KoCks"
access_token="808909816584409088-V9StBc9GXqHjKn9ueur08ZmDrtE4Z97"
access_token_secret="N81uFTvoa3dsFs41FklieP1hTHxTkKNFpTi8kdXKu6CpN"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
print("Connected to tweeter")

#Parameters
msg_limit=100000
language="thai"
hashtag=["ประเทศไทย"]

print("Stream started")

class CSVListener(StreamListener):
	def on_data(self,data):
		all_data=json.loads(data)
		if 'text' in all_data:
			tweet=all_data["text"]
			username=all_data["user"]["screen_name"]
			d.text=tweet
			d.user=username
			with open('python.json','a') as f:
				f.write(d)

			return True

		else:
			return True

stream=tweepy.Stream(auth=api.auth,listener=CSVListener())
iterator=stream.filter(track=['ตลก','ประเทศไทย'],async=True)

