"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import logging
import sys
import time

from cifar import cifar10
import numpy as np
import tensorflow as tf

from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session

# Base name for the checkpoint files
CHECKPOINT_BASENAME = 'model.ckpt'


def parse_args(unparsed_args=None):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--checkpoint_interval_steps',
      default=1000,
      type=int,
      help='Number of global steps between checkpoints.')

  parser.add_argument(
      '--summary_interval_steps',
      default = 100,
      type=int,
      help='Number of global steps between summaries.')

  parser.add_argument(
      '--log_device_placement',
      dest='log_device_placement',
      action='store_true')

  parser.add_argument(
      '--no-log_device_placement',
      dest='log_device_placement',
      action='store_false')

  parser.set_defaults(log_device_placement=True)

  cifar10.add_basic_model_parameters(parser)

  # TODO(jlewi): We ignore unknown arguments because the backend is currently
  # setting some flags to empty values like metadata path.
  args, _ = parser.parse_known_args(args=unparsed_args)
  return args


def train(args):
  """Train CIFAR-10 for a number of steps.

  Args:
    args: The command line arguments.
  """

  with tf.Graph().as_default():

    # Create the global step
    global_step = tf.contrib.framework.create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs(args.data_dir, args.batch_size,
                                              args.use_fp16)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, args.batch_size, args.use_fp16)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, args.batch_size)

    scaffold = monitored_session.Scaffold()

    session_creator = monitored_session.ChiefSessionCreator(
      scaffold, checkpoint_dir=args.train_dir, config=tf.ConfigProto(
        log_device_placement=args.log_device_placement))

    hooks = [
          # Hook to save the model every N steps and at the end.
          basic_session_run_hooks.CheckpointSaverHook(
              args.train_dir, checkpoint_basename=CHECKPOINT_BASENAME,
              save_steps=args.checkpoint_interval_steps, scaffold=scaffold),

          # Hook to save a summary every N steps.
          basic_session_run_hooks.SummarySaverHook(
              save_steps=args.summary_interval_steps, output_dir=args.train_dir,
              scaffold=scaffold),

          # Hook to stop at step N.
          basic_session_run_hooks.StopAtStepHook(last_step=args.train_max_steps)
      ]

    # Start a new monitored session. This will automatically restart the
    # sessions if the parameter servers are preempted.
    with monitored_session.MonitoredSession(session_creator=session_creator,
      hooks=hooks) as sess:

      while not sess.should_stop():

        start_time = time.time()
        _, loss_value, global_step_value = sess.run([train_op, loss,
          global_step])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if global_step_value % 10 == 0:
          num_examples_per_step = args.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          logging.info(('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)'), datetime.now(), global_step_value,
                       loss_value, examples_per_sec, sec_per_batch)


def main(argv=None):  # pylint: disable=unused-argument
  args = parse_args(argv)
  cifar10.maybe_download_and_extract(args.data_dir)
  if tf.gfile.Exists(args.train_dir):
    tf.gfile.DeleteRecursively(args.train_dir)
  tf.gfile.MakeDirs(args.train_dir)
  train(args)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main(sys.argv)
