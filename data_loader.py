from __future__ import unicode_literals
import numpy as np
import re
import codecs
from random import shuffle

genre_vectors = {
  'folk-country':   [1, 0, 0, 0, 0, 0, 0, 0],
  'electronica':    [0, 1, 0, 0, 0, 0, 0, 0],
  'metal':          [0, 0, 1, 0, 0, 0, 0, 0],
  'pop':            [0, 0, 0, 1, 0, 0, 0, 0],
  'dance':          [0, 0, 0, 0, 1, 0, 0, 0],
  'punk':           [0, 0, 0, 0, 0, 1, 0, 0],
  'rap':            [0, 0, 0, 0, 0, 0, 1, 0],
  'rock':           [0, 0, 0, 0, 0, 0, 0, 1]
}

genre_ids = {
  0:  'folk-country',
  1:  'electronica',
  2:  'metal',
  3:  'pop',
  4:  'dance',
  5:  'punk',
  6:  'rap',
  7:  'rock'
}

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def map_genre(genre):
    if genre.find('christian') > -1:
        return 'christian'
    elif genre.find('blues') > -1:
        return 'blues'
    elif genre.find('boogie') > -1:
        return 'blues'
    elif genre.find('experimental') > -1:
        return 'experimental'
    elif genre.find('pop') > -1:
        return 'pop'
    elif genre.find('punk') > -1:
        return 'punk'
    elif genre.find('house') > -1:
        return 'rap'
    elif genre.find('metal') > -1:
        return 'metal'
    elif genre.find('rap') > -1:
        return 'rap'
    elif genre.find('hip hop') > -1:
        return 'rap'
    elif genre.find('jazz') > -1:
        return 'jazz'
    elif genre.find('folk') > -1:
        return 'folk-country'
    elif genre.find('disco') > -1:
        return 'dance'
    elif genre.find('country') > -1:
        return 'folk-country'
    elif genre.find('bluegrass') > -1:
        return 'folk-country'
    elif genre.find('swing') > -1:
        return 'jazz'
    elif genre.find('skool') > -1:
        return 'rap'
    elif genre.find('electro') > -1:
        return 'electronica'
    elif genre.find('dance') > -1:
        return 'dance'
    elif genre.find('synth') > -1:
        return 'electronica'
    elif genre.find('grunge') > -1:
        return 'rock'
    elif genre.find('alternative rock') > -1:
        return 'rock'
    elif genre.find('indie rock') > -1:
        return 'rock'
    elif genre.find('thrash') > -1:
        return 'metal'
    elif genre.find('goth') > -1:
        return 'metal'
    elif genre.find('classic') > -1:
        return 'classical'
    elif genre.find('british invasion') > -1:
        return 'rock'
    elif genre.find('trance') > -1:
        return 'dance'
    elif genre.find('core') > -1:
        return 'dance'
    elif genre.find('rock') > -1:
        return 'rock'
    elif genre.find('death') > -1:
        return 'metal'
    elif genre.find('garage') > -1:
        return 'rock'
    elif genre.find('alternative') > -1:
        return 'rock'
    elif genre.find('funk') > -1:
        return 'funk'
    elif genre.find('reggae') > -1:
        return 'funk'
    elif genre.find('ska') > -1:
        return 'funk'
    elif genre.find('dub') > -1:
        return 'funk'
    elif genre.find('orchestral') > -1:
        return 'classical'
    elif genre.find('shoegazing') > -1:
        return 'rock'
    elif genre.find('hip-hop') > -1:
        return 'rap'
    elif genre.find('techno') > -1:
        return 'dance'
    elif genre.find('psychedel') > -1:
        return 'rock'
    elif genre.find('rave') > -1:
        return 'dance'
    return None

def clean_data(max_docs_per_genre = 2000):
  lines = codecs.open('raw-data.txt', 'r', encoding='utf-8').readlines()
  albumGneres = map(lambda x: x.split('\t'), lines)
  albumGneres = map(lambda x: (x[0], x[1].lower().replace(',', '').strip()), albumGneres)
  shuffle(albumGneres)
  genres = map(lambda x: x[1], albumGneres)
  genresDic = {}.fromkeys(genres)

  mappedGenres = {}
  for genre in genresDic:
    mapped = map_genre(genre)
    if mapped is not None:
        mappedGenres[genre] = mapped

  stats = {}
  f = codecs.open('data.txt', 'w', encoding='utf-8')
  for entry in albumGneres:
    genre = entry[1]
    if genre in mappedGenres:
      if mappedGenres[genre] not in stats:
        stats[mappedGenres[genre]] = 0
      stats[mappedGenres[genre]] += 1

      # choose only important genres and only up to max_docs_per_genre
      if mappedGenres[genre] in genre_vectors and stats[mappedGenres[genre]] <= max_docs_per_genre:
        f.write('{}\t{}\n'.format(entry[0], mappedGenres[genre]))
  f.close()

def get_data():
  lines = codecs.open('data.txt', 'r', encoding='utf-8').readlines()
  albumGneres = map(lambda x: x.split('\t'), lines)
  albumGneres = map(lambda x: (x[0], x[1].strip()), albumGneres)
  albums = map(lambda x: x[0], albumGneres)
  genres = map(lambda x: x[1], albumGneres)
  return albumGneres, albums, genres

def load_data_and_labels():
  """
  Loads MR polarity data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  """
  # Load data from files

  albumGneres, albums, genres = get_data()

  x_text = [album for album in albums]
  x_text = [clean_str(sent) for sent in x_text]

  # Generate labels
  labels = [genre_vectors[_] for _ in genres]

  return [x_text, labels]
