"""Pipeline for training a model that can read time from clock images.

This sets up a model and loss function, runs the optimization, and prints
performance statistics on a regular basis. It also saves the model and summaries
to disk (so they can be visualized in Tensorboard).

This pipeline is strongly inspired by the cifar10 pipeline.

 - The data is loaded from clock_data.py
 - The model is built in clock_model.py
 - For evaluating the trained model, see clock_evaluation.py

"""

import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import os.path

import clock_model
import clock_data


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tf_data',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 800,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train(summary_path):
    """ Builds and trains the clock reading model. """
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, (labels_hours, labels_minutes), num_records, num_classes = \
            clock_data.load_inputs_both(
                batch_size=FLAGS.batch_size, filename='clocks_train.txt')

        tf.image_summary("images/input", images)  # Visualize some input clocks.

        print('Training on {} images.'.format(num_records))
        print('Saving output to {}'.format(summary_path))

        # Build a Graph that computes the logits predictions from the
        # inference model in a multi-task learning .
        (logits_hours, logits_minutes) = clock_model.inference_multitask(images)
        logits = (logits_hours, logits_minutes)

        # Calculate loss.
        loss = clock_model.loss_multitask(logits_hours, labels_hours,
                                          logits_minutes, labels_minutes)

        # Compute accuracy (how often prediction is correct) for both minutes
        # and hours separately. in_top_k returns how often the true label is in
        # the top k of the predictions, k = 1 means only a match gets counted.
        train_accuracy_h_op = tf.nn.in_top_k(logits_hours, labels_hours, 1)
        train_accuracy_m_op = tf.nn.in_top_k(logits_minutes, labels_minutes, 1)

        # This operation computes the actual time error in minutes (i.e. how far
        # off our prediction was).
        time_error_losses = clock_model.time_error_loss(
            logits_hours, logits_minutes, labels_hours, labels_minutes)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = clock_model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Loss and timing statistics.
            if step % 20 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            # Compute **training** set precision and time error.
            if step % 30 == 0:
                # Evaluate the precision of all the top-k operators.
                precisions, total_count = clock_model.evaluate_precision(
                    sess, coord, num_records, FLAGS.batch_size,
                    [train_accuracy_h_op, train_accuracy_m_op])
                precision_h, precision_m = precisions

                print('%s: training set precision = %.3f(h) %.3f(m) \t '
                      '(%d samples)' % (datetime.now(), precision_h,
                                        precision_m, total_count))

                # Add it to summary writer.
                precision_summary_h = tf.scalar_summary(
                    'training_precision/hours', precision_h)
                precision_summary_m = tf.scalar_summary(
                    'training_precision/minutes', precision_m)
                precision_summary_c = tf.scalar_summary(
                    'training_precision/combined',
                    (precision_h + precision_m) * 0.5)

                # Compute time error in minutes.
                (time_err_c, time_err_h, time_err_m) = sess.run(
                    time_error_losses)

                print('%s: training set time error = %.3fm (total) \t'
                      ' %.3f(h) %.3f(m)'
                      % (datetime.now(), time_err_c, time_err_h, time_err_m))
                time_summary_c = tf.scalar_summary(
                    'training_error/combined', time_err_c)
                time_summary_h = tf.scalar_summary(
                    'training_error/hours_only', time_err_h)
                time_summary_m = tf.scalar_summary(
                    'training_error/minutes_only', time_err_m)

                summaries = sess.run([precision_summary_c, precision_summary_h,
                                      precision_summary_m, time_summary_c,
                                      time_summary_h, time_summary_m])
                for summary in summaries:
                    summary_writer.add_summary(summary, global_step=step)

            # Run summary writers for tensorboard
            if step % 20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 25 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(summary_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print('%s: saved model at step %d' % (datetime.now(), step))

        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument

    time_str = time.strftime('%H.%M.%S')
    summary_path = os.path.join(FLAGS.train_dir, 'run_{}'.format(time_str))

    tf.gfile.MakeDirs(summary_path)
    train(summary_path)


if __name__ == '__main__':
    tf.app.run()
