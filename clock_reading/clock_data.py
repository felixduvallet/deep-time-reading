""" Load images of clock faces.

We load a list of images specified in a file that has the following format:
    path/to/image1.png    HH    MM
    path/to/image2.png    HH    MM
(where HH and MM are hours and minutes).

Then, this sets up three batch queues: one for the image, one for the hour
label (integer), and one for the minute label (integer).

Each queue is randomized, and repeatedly samples from the master file (with
replacement). Sampling a queue gives a single batch of examples (the batch size
is specified as an input).

"""

import numpy as np
import tensorflow as tf

image_size = 57

# One channel = grayscale.
image_channels = 1


def read_labeled_image_list(image_list_file):
    """
    Reads a .txt file containing paths, hours, and minutes in three columns.
    """
    f = open(image_list_file, 'r')
    examples = []
    for line in f:
        # Convert tabs to spaces (of course) and remove \n characters.
        line = line.replace('\t', ' ')
        line = line.replace('\n', '')
        examples.append(line)
    return examples


def read_image_and_label(image_label_q):
    # Returns three Tensors: the decoded PNG image, the hour, and the minute.
    filename, hour_str, minute_str = tf.decode_csv(
        image_label_q.dequeue(), [[""], [""], [""]], " ")
    file_contents = tf.read_file(filename)

    # Decode image from PNG, and cast it to a float.
    example = tf.image.decode_png(file_contents, channels=image_channels)
    image = tf.cast(example, tf.float32)

    # Set the tensor size manually from the image.
    image.set_shape([image_size, image_size, image_channels])

    # Do per-image whitening (zero mean, unit standard deviation). Without this,
    # the learning algorithm diverges almost immediately because the gradient is
    # too big.
    image = tf.image.per_image_whitening(image)

    # The label should be an integer.
    hour = tf.string_to_number(hour_str, out_type=tf.int32)
    minute = tf.string_to_number(minute_str, out_type=tf.int32)

    return image, hour, minute


def setup_inputs(batch_size, fname='clocks.txt'):
    """ Get *all* inputs: the images, the hours, and the minutes. """
    combined_strings = read_labeled_image_list(fname)
    num_records = len(combined_strings)
    combined_queue = tf.train.string_input_producer(combined_strings)
    img, hour, minute = read_image_and_label(combined_queue)

    # Batch up training examples (images and labels).
    img_batch, hour_batch, minute_batch = tf.train.shuffle_batch(
        [img, hour, minute],
        batch_size=batch_size, num_threads=1,
        capacity=100, min_after_dequeue=10)

    return img_batch, hour_batch, minute_batch, num_records


def load_inputs_hours(batch_size, filename):
    img_batch, hour_batch, minute_batch, num_records = setup_inputs(
        batch_size, fname=filename)
    num_classes = 12
    return img_batch, hour_batch, num_records, num_classes


def load_inputs_minutes(batch_size, filename):
    img_batch, hour_batch, minute_batch, num_records = setup_inputs(
        batch_size, fname=filename)
    num_classes = 60
    return img_batch, minute_batch, num_records, num_classes


def load_inputs_both(batch_size, filename):
    # This is useful for multitask learning.
    img_batch, hour_batch, minute_batch, num_records = setup_inputs(
        batch_size, fname=filename)

    num_classes = (60, 12)
    return img_batch, (hour_batch, minute_batch), num_records, num_classes


def load_inputs(batch_size, filename, output_type):
    # Parameter-switched version of the above methods.
    if output_type is 'minutes':
        return load_inputs_minutes(batch_size, filename)
    elif output_type is 'hours':
        return load_inputs_hours(batch_size, filename)
    else:
        raise(TypeError('Invalid output type: {}'.format(output_type)))


def run_wholefile():
    # This is a very simple example of using the batch queues.

    img_batch, label_batch, minute_batch, num_records = setup_inputs(
        batch_size=8)

    print('Loaded queues from {} examples.'.format(num_records))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            count = 0
            while not coord.should_stop():

                # Get a pair of image and label lists. NOTE that as far as I can
                # tell, the queues are always synchronized (you can 'run' the
                # label queue and still get 'valid' pairs of image/labels
                # afterwards).
                (img_eval, label_eval) = sess.run([img_batch, label_batch])

                # Just print out the image label and the sum of its pixel
                # values (this is a kind of 'identifier' we can use to see which
                # image we have loaded).
                img_sums = [np.sum(x) for x in img_eval]
                for (img, label) in zip(img_sums, label_eval):
                    print('{} \t {}'.format(label, img))

                count += 1
                if count > 10:
                    break
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    run_wholefile()
