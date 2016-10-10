Deep time reading
-----------------

Using deep learning to tell time from images of clocks with Tensorflow.

This is a side-project I decided to start to learn a bit more about Tensorflow's
architecture and data flow pipeline.
It is not meant to be a production-level solution, this is a quick-and-dirty
"demo" of Tensorflow that learns quickly, so we can evaluate many models and
experiment with different features of tensorflow.
Deep learning is **definitely** overkill for this problem (especially with the
simple data we will use), but again the point is to have a sandbox to experiment
with tensorflow's features.

Contributions are welcome via issues and pull requests.

## Basic idea

Learn to tell time from an image of a clock, like this one:

![alt text][clock]

[clock]: https://raw.githubusercontent.com/felixduvallet/deep-time-reading/master/assets/clock-10.10.00.png "Clock face"

The solution in a nutshell will be:
 * For input data, we will generate a synthetic clock face with different times shown.
 * The input pipeline will use tensorflow batch queues to provide a parallelizable workflow.
 * We will treat this problem as a multi-task classification problem, where one
   task is to predict the hour and one task is to predict the minute.
 * The model architecture is essentially the same as CIFAR10's.
 * The loss function optimized will be cross entropy.
 * We will also measure the time error (how far off our time telling is from the
   truth).

## Generating clocks

While it would be awesome to go out and get a bunch of pictures of clocks and
label them with their time, that would make a sizeable project in its own right
(probably involving existing image databases, some cropping, and Amazon
Mechanical Turk).

Instead, I decided to use synthetic data (clock faces), because it's easy to
generate and by definition annotated with the correct time.

The code for generating arbitrary clocks is in `generate_clocks.py`, and uses
matplotlib.
Currently only hours and minutes and drawn, but seconds are supported if you
want to make the problem more complicated.
The script generates 57x57 images of clock faces with a time, for example 
10'10: ![alt text][clock]

[clock]: https://raw.githubusercontent.com/felixduvallet/deep-time-reading/master/assets/clock-10.10.00.png "Clock face"

So let's generate a bunch of clocks!

NOTE: Instructions for running code are shown like this:

> Run the code to generate a bunch of clocks for all times:

```
$ python clock_reading/generate_clocks.py
```

It will create a directory full of clock images, and also an index file
(`clocks_all.txt`), which lists the filename, hour, and minute for all generated
clocks.
If we generate a clock for every single minute over 12 hours, we get 720 clock
images and 720 lines.

We can now separate the 720 examples into independent train & test sets using
the `shuf` utility (NOTE: in OSX, install coreutils and use `gshuf`).

Let's take 80% of the data as training examples (576 images) and the rest as
test set:


    $ shuf clocks_all.txt > clocks_rand.txt  
    $ head -n 576 clocks_rand.txt > clocks_train.txt
    $ tail -n 144 clocks_rand.txt > clocks_test.txt

Now we have data (clock images) and an index file that provides the ground truth
label (for hours and minutes). We are ready to start learning.

## Deep learning using Tensorflow

We will treat this problem as a classification problem on hours and minutes
separately.
More concretely, the classifier will take an image and predict an integer
(between 0 and 11 for hours, or 0 and 59 for minutes).
Later, we will show how to combine these two predictors into one multi-task
model that can tell the complete time.

### Tensorflow input pipeline

`clock_data.py`


### Tensorflow model

For our computational model layout we will use basically the CIFAR10 model of
[Alex Krizhevsky] available in the tensorflow [codebase].

[Alex Krizhevsky]: http://www.cs.toronto.edu/~kriz/index.html
[codebase]: https://github.com/tensorflow/tensorflow/tree/r0.10/tensorflow/models/image/cifar10

First we'll make sure the CIFAR10 model is acceptable for a simple version of
this task (i.e. it produces accurate predictions).
Then, we'll build upon this model to predict the complete time from a clock
images.

The classifier will have several layers, and the output layer will be a
fully-connected linear softmax layer (i.e. multi-class logistic regression) that
produces a probability score.

The file `clock_model.py` specifies the inference graph, the loss function, and
the training operation (the optimization). 

#### 1. A simple classifier for hours or minutes

First, we should make sure CIFAR10's model is sufficient for detecting clock
hands and predicting the correct time. 
Given this is a relatively simple problem (the input data is very very good...
because we made it), we should expect this to be the case, but it's always good
to check.

We can train two models, one that takes an image and produces a 12-dimensional
probability distribution (for hours) and one that takes the same image and
produces a 60-dimensional distribution (for minutes).

NOTE: For code simplicity I decided not to keep the single-task model, as it was
going to be a pain to maintain every piece of code twice (once for predicting
time separately, once for a combined prediction).

#### 2. A multi-task classifier for time reading

Predicting hours *or* minutes using deep learning is pretty cool, but it's not
terribly useful.
While we could theoretically train a model for hours *and* a model for minutes,
this would be a lot of duplicated work.
However, if you think about it a lot of the problem structure (isolating clock
hands, finding their location, etc.) is shared across both problems.

We will thus leverage the concept of [multi-task learning], where one
computation graph is tasked with predicting two different outputs using (almost)
entirely the same structure.
All the layers are shared, except for the output layer which is duplicated: once
for hours and once for minutes.
In this way, the model uses the same parameters for all other layers.

[multi-task learning]: http://link.springer.com/article/10.1023/A:1007379606734

The final model looks something like this:

TODO: Add image of graph.

As you can see, the input is fed through several layers, then the final layer
diverges: one softmax classifier for hours and one for minutes.
Note that both classifiers are using the same learned features.

### Learn!

(NOTE: Make sure you have tensorflow installed and loaded.)

The file `clock_training.py` instantiates the model defined above, loads the
data queues, and handles a lot of book-keeping during training.

> We can now run the training pipeline 

    python clock_reading/clock_training.py 

This will load all the images (and labels) into a never-ending data pipeline (by
repeating the input), shuffle the order of the data, and train the model.

Every few iterations you will see the loss (cross entropy) for the current step,
as well as the prediction and time accuracies (on the training data).
We also save the latest model periodically, to later load it and evaluate the 
performance on **test** data.

You can also visualize the loss over time using tensorboard:

    tensorboard --logdir tf_data/ --reload_interval 15    

And finally navigate to <http://localhost:6006/> to see the learning progress.

NOTE: We store each run of tensorflow separately.
They will all show up in tensorboard.

### Evaluating the learned model

In the files `clock_model.py` and `clock_training.py`, we have several loss
functions defined to help us evaluate the performance of our system

Error metrics:

1. Cross-entropy loss:
    
    We can define a loss for each task (hours or minutes):

        def loss(logits, labels):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, labels, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    And combine them to end up with a cumulative classification loss:

        loss_hours = loss(logits_h, labels_h)
        loss_minutes = loss(logits_m, labels_m)
        tf.add_n([loss_hours, loss_minutes], name='loss')

    Note that this is the loss that *actually* gets optimized during training.
    Its aim is to maximize the prediction class accuracy, i.e. get as many hour and 
    minute predictions correct as possible.

2. Classification precision

    One useful metric is to know how often we get the classification correct
    (i.e. I predict minutes = 15 and it matches ground truth).
    
    We use tf's `in_top_k` with k=1 to do exactly that, for both hours and
    minutes.
    
        train_accuracy_h_op = tf.nn.in_top_k(logits_hours, labels_hours, 1)
        train_accuracy_m_op = tf.nn.in_top_k(logits_minutes, labels_minutes, 1)
    
    This is shown in the log as follows: 
    
        training set precision = 0.169(h) 0.034(m) 	 (640 samples)
      
    means we got 17% of the hours correct and 3% of the minutes correct.
    Note this does not account for *how far off* any prediction might be, being
    wrong by one minute is the same as being wrong by 30 minutes.

3. Time error
    
    Lastly, we can also compute a loss which measures how far off our predicted
    time is from the true time.
    We will compute this difference (expressed in minutes) in
    `clock_model.time_error_loss(-)`:

        avg_error_c = tf.reduce_mean(wrap(tf.sub(tf.add(60 * hours_predicted, minutes_predicted),
                            tf.add(60 * hours_true, minutes_true)))

    where `wrap(-)` just ensures we wrap around the clock when computing time
    differences, so that the time difference between 9'58 and 10'02 is 4
    minutes.

    Side note: Tensorflow's `mod` operator works differently than python's, so
    we have to do some more work to ensure we get the correct wraparound time
    difference operator.
    
#### Running the evaluation pipeline 

You'll notice that we have a separate evaluation pipeline.
This enables us to compute the validation set accuracy *during* the 
optimization, and see how well we are doing on a held-out set of clock images.


   1. Instantiate the same model as the training pipeline
   1. Set up a data pipeline on held-out validation (or test) data
   1. Then, every few seconds:
      1. Load the latest trained model file (sets the parameters)
      1. Run the model on the held out data
      1. Report metrics

> Run the evaluation pipeline

    $ python clock_reading/clock_evaluation.py

This will report the holdout accuracy every few seconds (few = 30 by default).

NOTE: When multiple runs are available in `tf_data/`, the evaluation loads the
latest one (ordered alphabetically).

> Visualize in tensorboard:

    $ tensorboard --port 6007 --logdir tf_eval/ --reload_interval 15
    
And navigate to <http://localhost:6007/>

## Observations & Notes

Training typically takes much longer in a multi-task setting because the model
has to find good features that represent *both* the hour and minute hands.

Performance is usually quite good here since it's a very simple problem: the
data is very clean and highly structured.
However, sometimes the model fails to converge to a good solution, even on the
training data.
In rare cases, it does overfit to the training data (performance is bad
only on the test dataset).
This is usually worse for the minutes prediction.

The model usually gets 100% precision on the training data (again, this is a 
very simple with very structured data so we expect quite good performance).
The test precision is not quite as high, but looking at individual samples shows
that when the model makes a mistake, it can often be just **off by a single
minute** (in rarer cases, off by an hour), as for example:

    Predicted time: 07:31. True time: 07:30
    Predicted time: 08:25. True time: 08:26
    Predicted time: 10:24. True time: 10:25

Not bad, deep learning, not bad.

## Future Work

Here are some suggestions for future work:

 - Train a regression model instead of classification.
 - Use an embedding projection to learn a lower-dimensional embedding of the
   clock images.
 

If any of these appeal to you, feel free to  fork this repo and submit a Pull
Request.
