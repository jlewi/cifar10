"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

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
import re
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
      '--num_gpus',
      default=1,
      type=int,
      help='Number of GPUs to use.')

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


def tower_loss(scope, args):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    args: Command line arguments.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  images, labels = cifar10.distorted_inputs(args.data_dir, args.batch_size,
                                            args.use_fp16)

  # Build a Graph that computes the logits predictions from the
  # inference model.
  logits = cifar10.inference(images, args.batch_size, args.use_fp16)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(args):
  """Train CIFAR-10 for a number of steps.

  Args:
    args: The command line arguments.
  """
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create the global step.
    global_step = tf.contrib.framework.create_global_step()

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             args.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(args.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss = tower_loss(scope, args)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # To understand why the following line is necessary, see:
    # https://github.com/carpedm20/DCGAN-tensorflow/issues/59
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      variable_averages = tf.train.ExponentialMovingAverage(
          cifar10.MOVING_AVERAGE_DECAY, global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    scaffold = monitored_session.Scaffold(summary_op=summary_op)

    # allow_soft_placement must be set to True to build towers on GPU, as some
    # of the ops do not have GPU implementations.
    session_creator = monitored_session.ChiefSessionCreator(
      scaffold, checkpoint_dir=args.train_dir, config=tf.ConfigProto(
        allow_soft_placement=True,
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
          num_examples_per_step = args.batch_size * args.num_gpus
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / args.num_gpus

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          logging.info(format_str % (datetime.now(), global_step_value,
            loss_value, examples_per_sec, sec_per_batch))


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

