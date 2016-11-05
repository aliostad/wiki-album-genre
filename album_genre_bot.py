import twitter
import eval_single
import click
import re
import os
from datetime import datetime, timedelta

timestamp_filename = '__last_timestamp__'
twitterId = '@AlbumGenreBot'

def get_time_stamp():

  if os.path.exists(timestamp_filename):
    return datetime.strptime(open(timestamp_filename, mode='r').read(), '%Y-%m-%d %H:%M:%S')

  else:
    return datetime.now() - timedelta(days=7)

def save_time_stamp(dt):
  open(timestamp_filename, 'w').write(dt.strftime('%Y-%m-%d %H:%M:%S'))

def clean_text(tweet_mention_text):
  return re.sub('@\w+', '', tweet_mention_text).strip()

def predict(text):
  return eval_single.get_genre(text)

def parse_twitter_date(s):
  return datetime.strptime(s, '%a %b %d %H:%M:%S +0000 %Y')

@click.command()
@click.option('--consumer_key', help='Twitter consumer key')
@click.option('--consumer_secret', help='Twitter consumer secret')
@click.option('--token_key', help='Twitter token key')
@click.option('--token_secret', help='Twitter token secret')
def respond(consumer_key, consumer_secret, token_key, token_secret):
  if consumer_key is None:
    print 'usage: python album_genre_bot.py <consumer_key> <consumer_secret> <token_key> <token_secret>'
    return

  api = twitter.Api(consumer_key=consumer_key,
                      consumer_secret=consumer_secret,
                      access_token_key=token_key,
                      access_token_secret=token_secret)

  original_timestamp = get_time_stamp()
  new_timestamp = original_timestamp
  for m in api.GetMentions():
    dt = parse_twitter_date(m.created_at)
    new_timestamp = max(dt, new_timestamp)
    if dt > original_timestamp and m.text.startswith(twitterId):
      genre = predict(clean_text(m.text).lower())
      try:
        status = api.PostUpdate(
          in_reply_to_status_id=m.id,
          status='@{} {}'.format(m.user.screen_name, genre))
        print m.text,
        print ' ==> ' + genre
      except Exception as e:
        print e

  save_time_stamp(new_timestamp)

if __name__ == '__main__':
  print respond()






