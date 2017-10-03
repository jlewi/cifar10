"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.
Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.
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
from datetime import timedelta
import logging
import math
import sys
import time

from cifar import cifar10
import numpy as np
import tensorflow as tf


def parse_args(unparsed_args=None):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser()

  #*****************************************************************************
  # args required by Cloud ML
  parser.add_argument(
      '--eval_dir',
      default=None,
      help='Directory where to write event logs for evaluation.')

  parser.add_argument(
      '--eval_data',
      default='test',
      help='Either "test" or "train_eval"')

  parser.add_argument(
      '--eval_interval_secs',
      default=60 * 5,
      type=int,
      help='How often to run the eval.')

  parser.add_argument(
      '--eval_num_examples',
      default=10000,
      type=int,
      help='Number of examples to run for evaluation.')

  parser.add_argument(
      '--run_once',
      dest='run_once',
      action='store_true')

  parser.add_argument(
      '--no-run_once',
      dest='run_once',
      action='store_false')

  parser.add_argument(
      '--max_wait_for_new_checkpoints_secs',
      default=60 * 20,
      help=('Maximum time to wait for a new checkpoint before giving up. '
            'This should be larger than eval_interval_secs'))

  parser.set_defaults(run_once=False)

  cifar10.add_basic_model_parameters(parser)

  # TODO(jlewi): We ignore unknown arguments because the backend is currently
  # setting some flags to empty values like metadata path.
  args, _ = parser.parse_known_args(args=unparsed_args)
  return args


def eval_once(saver, summary_writer, top_k_op, summary_op, checkpoint_dir,
              num_examples, batch_size):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    checkpoint_dir: The directory containing the model checkpoints
    num_examples: Number of examples to use for evaluation.
    batch_size: The batch size.

  Returns:
    global_step: The global step of the checkpoint evaluated.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      global_step = int(global_step)
    else:
      logging.info('No checkpoint file found')
      return
    logging.info('Processing checkpoint for global step: %s', global_step)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(
            qr.create_threads(
                sess, coord=coord, daemon=True, start=True))

      num_iter = int(math.ceil(num_examples / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      # The '1' here refers to k in tf.nn.in_top_k.
      precision = true_count / total_sample_count
      logging.info('%s: global step: %s precision @ 1 = %.3f', datetime.now(),
                   global_step, precision)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step


def evaluate(args):
  """Eval CIFAR-10 for a number of steps."""
  if args.max_wait_for_new_checkpoints_secs <= args.eval_interval_secs:
    raise ValueError('max_wait_for_new_checkpoints_secs should be > '
                     'eval_interval_secs')
  max_wait_for_new_checkpoints = timedelta(
      seconds=args.max_wait_for_new_checkpoints_secs)

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = args.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data,
                                    data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    use_fp16=args.use_fp16)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, args.batch_size, args.use_fp16)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(args.eval_dir, g)

    # TODO(jlewi): Could we detect convergence and abort early?
    # TODO(jlewi): eval_once always picks up the latest checkpoints. So we
    # could potentially skip intermediate checkpoints if we can't keep up with
    # the checkpoints produced.
    global_step = 0
    logging.info('Evaluation expects %s training steps.', args.train_max_steps)

    # Keep track of the time we last found a new checkpoint.
    # We use this to avoid deadlocking in the event no new checkpoints are
    # being produced.
    time_last_checkpoint = datetime.now()
    while (global_step + 1) < args.train_max_steps:
      # latest_global_step will be None if no checkpoint is found.
      latest_global_step = eval_once(
          saver, summary_writer, top_k_op, summary_op, args.train_dir,
          args.eval_num_examples, args.batch_size)
      old_global_step = global_step
      global_step = max(latest_global_step, global_step)
      if global_step != old_global_step:
        time_last_checkpoint = datetime.now()
      if args.run_once:
        logging.info('Exiting because a single evaluation was requested.')
        break

      if datetime.now() - time_last_checkpoint > max_wait_for_new_checkpoints:
        message = ('No new checkpoints were found in the last {0} minutes; '
                   'giving up.').format(
                       max_wait_for_new_checkpoints.total_seconds() / 60.0)
        logging.error(message)
        raise RuntimeError(message)
      time.sleep(args.eval_interval_secs)
    logging.info('Finished evaluating checkpoints.')


def main(argv=None):  # pylint: disable=unused-argument
  args = parse_args(argv)
  cifar10.maybe_download_and_extract(args.data_dir)
  if tf.gfile.Exists(args.eval_dir):
    tf.gfile.DeleteRecursively(args.eval_dir)
  tf.gfile.MakeDirs(args.eval_dir)
  evaluate(args)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main(sys.argv)
