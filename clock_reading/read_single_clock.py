""" Read the time from a single image.

This runs the trained tensorflow model with a single image as the input.

It also shows the top 3 matches (for both hours and minutes), and their
respective probabilities.

It can either use the image specified by the hour/minutes (assuming the same
file naming scheme as the rest of this project), or you can pass it a filename
directly.

"""

import tensorflow as tf
import numpy as np

from clock_reading import clock_model
from clock_reading import clock_evaluation
from clock_reading import clock_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('hour', 1,
                            """Ground truth hour.""")
tf.app.flags.DEFINE_integer('minute', 5,
                            """Ground truth minute.""")
tf.app.flags.DEFINE_string('image', None,
                           """Image file (optional).""")


def main(hour, minute, fname=None):
    # ** Read the data. **
    true_h, true_m = hour, minute

    # Create a fake queue object so that TF's image reader will get a
    # filename and label when it calls dequeue().
    class FakeQueue(object):
        def dequeue(self):

            fpath = fname  # Use the filepath provided, otherwise make it up.
            if not fpath:
                fpath = 'clocks/clock-{:02d}.{:02d}.00.png'.format(
                    true_h, true_m)
            print('Reading image: {}'.format(fpath))

            stuff = '{} \t{} \t{}'.format(fpath, true_h, true_m)
            return stuff

    q = FakeQueue()
    image, hour, minute = clock_data.read_image_and_label(q)
    image = tf.reshape(image, [1, 57, 57, 1])  # Reshape into a batch of one.

    # ** Build the model. **

    # The model is expressed in log probabilities, softmax gets the true class
    # probabilities.
    (logits_hours, logits_minutes) = clock_model.inference_multitask(image)
    likelihood_h = tf.nn.softmax(logits_hours)
    likelihood_m = tf.nn.softmax(logits_minutes)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        clock_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:

        # Load the saved model from file.
        global_step = clock_evaluation.load_model(sess, saver)
        if global_step is None:
            return

        # Evaluate the model; get probabilities.
        ([prob_h], [prob_m], label_h, label_m) = sess.run(
            [likelihood_h, likelihood_m, hour, minute])

        # Sort in descending order.
        sort_idx_h = np.argsort(prob_h)[::-1]
        sort_idx_m = np.argsort(prob_m)[::-1]

        print('==================')
        print('Top 3 predictions:')

        for idx in range(0, 3):
            (pred_h, pred_m) = sort_idx_h[idx], sort_idx_m[idx]
            correct_h = '*' if pred_h == label_h else ' '
            correct_m = '*' if pred_m == label_m else ' '

            print('  H {:2d} (p = {:.2f}) {} |  M {:2d} (p = {:.2f})  {}'.format(
                pred_h, prob_h[pred_h], correct_h,
                pred_m, prob_m[pred_m], correct_m,
            ))
        print('==================')
        print('Truth: H {:2d}  |  M {:2d}'.format(label_h, label_m))


if __name__ == "__main__":

    main(hour=FLAGS.hour,
         minute=FLAGS.minute,
         fname=FLAGS.image)
