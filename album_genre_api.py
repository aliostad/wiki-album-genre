#! /usr/bin/env python
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_loader
from text_cnn import TextCNN
from tensorflow.contrib import learn
from flask import Flask, jsonify, request, send_from_directory
import logging
from logging import FileHandler
import time

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

FLAGS.checkpoint_dir = './runs/1477424694/checkpoints/' # 1477424694 1477602229
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()


t = time.time()
if __name__ == '__main__':
  print("Time is now {}".format(t))

app = Flask(__name__)
application = app

try:
  file_handler = FileHandler("album_genre_api.log", "a")
  file_handler.setLevel(logging.WARNING)
  app.logger.addHandler(file_handler)

  if __name__ == '__main__':
    print("Took {}".format(time.time() - t))
except:
  if __name__ == '__main__':
    print("could not start file logging")

@app.route('/', methods=['GET'])
def browse_default():
  try:
    return send_from_directory('ui', 'index.html')
  except Exception as e:
    return e.message

@app.route('/<path:path>', methods=['GET'])
def staticx(path):
   return send_from_directory('ui', path)


@app.route('/api/v1/album/genre', methods=['GET'])
def get_genre():

  try:

    with graph.as_default():

      session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        albums = request.args.get('albums')
        x_raw = albums.split(',')
        all_predictions = []
        x_test = np.array(list(vocab_processor.transform(x_raw)))

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_loader.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        return jsonify({'results': map(lambda x: data_loader.genre_ids[int(x)], all_predictions)})

  except Exception as e:
    print e
