"""Entrypoint for Cloud ML jobs.

The point of this entrypoint is to invoke the evluation or training code
depending on the task.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import sys

from cifar import cifar10_eval
from cifar import cifar10_multi_gpu_train
from cifar import cifar10_train
import tensorflow as tf


def parse_args(unparsed_args=None):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--num_gpus',
      default=0,
      type=int,
      help='Number of GPUs to use.')

  parser.set_defaults(log_device_placement=True)

  args, _ = parser.parse_known_args(args=unparsed_args)
  return args


def main(argv=None):  # pylint: disable=unused-argument
  args = parse_args(argv)
  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  logging.info('Started with config: %s', config)
  logging.info('Tensorflow version: %s', tf.__version__)
  logging.info('Tensorflow git version: %s', tf.__git_version__)
  task_type = config.get('task', {}).get('type', None)
  if task_type == 'master':
    logging.info('Running as master; running eval.')
    cifar10_eval.main()
  elif task_type == 'worker':
    logging.info('Running as worker; running training.')
    if args.num_gpus > 1:
      logging.info('Running training with multi GPU support.')
      cifar10_multi_gpu_train.main()
    else:
      cifar10_train.main()
  else:
    logging.error('Unrecognized task type %s', task_type)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main(sys.argv)
