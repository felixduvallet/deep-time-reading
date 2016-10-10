"""Evaluation pipeline for reading time from clock images.

Run this pipeline separately from the clock_training pipeline, and it will
evaluate the performance of the latest trained model available. You can run
this in parallel to see how the test-set performance improves per iterations.

By default it evaluates the model every few seconds (eval_interval_secs), but
you can also evaluate once and quit.

It saves the output to a separate summary directory, so you can run another
Tensorboard instance.

This pipeline is strongly inspired by the cifar10 pipeline:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image/cifar10

"""
from __future__ import division
from __future__ import print_function

import math
import time
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from clock_reading import clock_model
from clock_reading import clock_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tf_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tf_data',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def find_model_dir(base_dir):
    # Get the latest model checkpoint folder from a bunch of date-named folders
    # inside the base directory.

    directories = tf.gfile.ListDirectory(base_dir)
    if not directories:
        return None
    latest = directories[-1]

    return os.path.join(base_dir, latest)


def load_model(session, saver):
    ckpt_dir = find_model_dir(FLAGS.checkpoint_dir)
    print('Trying to load model from: {}...'.format(ckpt_dir))

    # Get directory for latest run inside checkpoints folder: run_HH.MM.SS.
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(session, ckpt.model_checkpoint_path)
        # Get global step from filename of the form model.ckpt-NNNN.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Loaded saved model from step {}'.format(global_step))
        return global_step
    else:
        print('No checkpoint file found, cannot load model.')
        return None


def eval_aggregate(saver, summary_writer, top_k_ops, num_records,
                   models, labels):
    """ Evaluate all samples in aggregate, compute statistics.
    """
    with tf.Session() as sess:

        global_step = load_model(sess, saver)
        if global_step is None:
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            # This is the classification accuracy (how often do we get classes
            # correct).
            precisions, total_count = clock_model.evaluate_precision(
                sess, coord, num_records, FLAGS.batch_size, top_k_ops)
            precision_h, precision_m = precisions

            print('%s: Test set precision = %.3f(h) %.3f(m) \t '
                  '(%d samples)' % (datetime.now(), precision_h,
                                    precision_m, total_count))

            # This is the actual time error (how many minutes off we are from
            # the truth).
            time_error_losses = clock_model.time_error_loss(
                models[0], models[1], labels[0], labels[1])
            (time_err_c, time_err_h, time_err_m) = sess.run(time_error_losses)

            print('%s: Test set time error = %.3fm (combined) \t'
                  ' %.3f(h) %.3f(m)'
                  % (datetime.now(), time_err_c, time_err_h, time_err_m))

            # Add everything to the summary writer.
            precision_summary_h = tf.scalar_summary(
                'test_precision/hours', precision_h)
            precision_summary_m = tf.scalar_summary(
                'test_precision/minutes', precision_m)
            precision_summary_c = tf.scalar_summary(
                'test_precision/combined',
                (precision_h + precision_m) * 0.5)

            time_summary_c = tf.scalar_summary(
                'test_error/combined', time_err_c)
            time_summary_h = tf.scalar_summary(
                'test_error/hours_only', time_err_h)
            time_summary_m = tf.scalar_summary(
                'test_error/minutes_only', time_err_m)

            summaries = sess.run(
                [precision_summary_c, precision_summary_h, precision_summary_m,
                 time_summary_c, time_summary_h, time_summary_m])
            for summary in summaries:
                summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def eval_samples(saver, summary_writer, models, labels):
    # Evaluate individual samples and print their predictions.
    with tf.Session() as sess:

        global_step = load_model(sess, saver)
        if global_step is None:
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
                pass

            predicted_times, true_times, sample_count = \
                clock_model.compute_time_predictions(
                    sess, coord, models, labels, num_records=FLAGS.batch_size,
                    batch_size=FLAGS.batch_size)
            time_errors = compute_time_errors(predicted_times, true_times)

            # This is the actual time error (how many minutes off we are from
            # the truth).
            time_error_losses = clock_model.time_error_loss(
                models[0], models[1], labels[0], labels[1])
            (time_err_c, time_err_h, time_err_m) = sess.run(time_error_losses)

            print('%s: Test set time error = %.3fm (combined) \t'
                  ' %.3f(h) %.3f(m)'
                  % (datetime.now(), time_err_c, time_err_h, time_err_m))


            print('Showing mistakes only:')
            correct_count = 0

            for (idx, (p, t)) in enumerate(zip(predicted_times, true_times)):
                if p == t:  # Don't show correct predictions.
                    correct_count += 1
                    continue

                print('   Predicted {:02d}\'{:02d} -- Actual {:02d}\'{:02d} '
                      '\t Error: {:.2f}'.format(
                    p[0], p[1], t[0], t[1], time_errors[idx, 0]))
            print('Skipped {} correct examples'.format(correct_count))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    pass


def compute_precision(predicted_times, true_times):
    """
    Compute percentage of exactly correct times.
    :param predicted_times:
    :param true_times:
    :return: float, percentage of times that are exactly correct.
    """

    correct = 0
    for (predicted, true) in zip(predicted_times, true_times):
        if predicted == true:
            correct += 1
    correct_percentage = float(correct) / len(predicted_times)

    return correct_percentage


def compute_time_errors(predicted_times, true_times):
    """
    Compute the time-telling error. We compute the aggregate error (expressed
    in minutes), but also the number of hours and minutes separately.

    :param predicted_times:
    :param true_times:
    :return: N x 3 np array, where each row is
    [total_error_in_minutes, hours_error, minute_error].
    """

    errors = np.zeros((len(predicted_times), 3))
    for (idx, (predicted, true)) in enumerate(zip(predicted_times, true_times)):

        time_predicted = 60 * predicted[0] + predicted[1]
        time_real = 60 * true[0] + true[1]
        delta_t = time_predicted - time_real

        delta_h = predicted[0] - true[0]
        delta_m = predicted[1] - true[1]

        # Account for wraparound times.
        errors[idx, 0] = min(delta_t % 720, -delta_t % 720)

        errors[idx, 1] = min(delta_h % 12, -delta_h % 12)
        errors[idx, 2] = min(delta_m % 60, -delta_m % 60)
    return errors


def evaluate(summary_path):
    """ Periodically evaluate the latest-available model.

    Sets up the same graph (defined in clock_model), then periodically searches
    for (and loads) the latest available
    """
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, (labels_hours, labels_minutes), num_records, num_classes = \
            clock_data.load_inputs_both(
                batch_size=FLAGS.batch_size, filename='clocks_test.txt')
        print('Loaded {} test images.'.format(num_records))

        # Build a Graph that computes the logits predictions from the
        # inference model.
        print('Building model...')
        (logits_hours, logits_minutes) = clock_model.inference_multitask(images)

        # Calculate whether prediction is correct or not.
        top_k_op_h = tf.nn.in_top_k(logits_hours, labels_hours, 1)
        top_k_op_m = tf.nn.in_top_k(logits_minutes, labels_minutes, 1)
        top_k_ops = [top_k_op_h, top_k_op_m]

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            clock_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(summary_path, g)

        # Run the evaluation every few seconds.
        while True:

            do_samples = False
            do_samples = True

            do_aggregate = False
            do_aggregate = True

            if do_samples:
                eval_samples(saver, summary_writer, (logits_hours, logits_minutes), (labels_hours, labels_minutes))
            if do_aggregate:
                eval_aggregate(saver, summary_writer, top_k_ops, num_records,
                               (logits_hours, logits_minutes),
                               (labels_hours, labels_minutes))

            if FLAGS.run_once:
                break
            print('{}: sleeping {} seconds'.format(
                datetime.now(), FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

    time_str = time.strftime('%H.%M.%S')
    summary_path = os.path.join(FLAGS.eval_dir, 'eval_{}'.format(time_str))

    tf.gfile.MakeDirs(summary_path)
    evaluate(summary_path)


if __name__ == '__main__':
    tf.app.run()

# Copyright 2015 The TensorFlow Authors and Felix Duvallet.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
