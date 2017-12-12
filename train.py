import timeit
import tensorflow as tf
from tensorflow.python.framework import ops
from model import Model, Phase


def train(config, train_batches, validation_batches, vocab_size, save_path, from_saved=False):
    print("Beginning training...")
    train_start_time = timeit.default_timer()

    train_batches, train_lens, train_labels, _ = train_batches
    validation_batches, validation_lens, validation_labels, _ = validation_batches

    if not from_saved:
        with open("training_history.tsv", "w") as history:
            history.write("epoch\t" +
                          "train_loss\t" +
                          "val_loss\t" +
                          "val_acc")
            history.close()

    ops.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        with tf.variable_scope("LSTM", reuse=False):
            train_model = Model(
                config=config,
                batch=train_batches,
                vocab_size=vocab_size,
                phase=Phase.Train)

        with tf.variable_scope("LSTM", reuse=True):
            validation_model = Model(
                config=config,
                batch=validation_batches,
                vocab_size=vocab_size,
                phase=Phase.Validation)

        # Initialize Saver
        saver = tf.train.Saver()

        if from_saved:
            print("LOADING FROM SAVED MODEL>>>>")
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, save_path)
            print("Model restored.")
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            epoch_start_time = timeit.default_timer()
            train_loss = 0.0
            validation_loss = 0.0
            train_accuracy = 0.0
            accuracy = 0.0

            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, acc, _ = sess.run([train_model.loss, train_model.accuracy, train_model.train_op], {
                    train_model.x: train_batches[batch], train_model.lens: train_lens[batch],
                    train_model.y: train_labels[batch]})
                train_loss += loss
                train_accuracy += acc

            # validation on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, acc = sess.run([validation_model.loss, validation_model.accuracy], {
                    validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch],
                    validation_model.y: validation_labels[batch]})

                validation_loss += loss
                accuracy += acc

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            train_accuracy /= train_batches.shape[0]
            accuracy /= validation_batches.shape[0]

            print(
                "epoch %d - train loss: %.3f, train acc: %.3f, validation loss: %.3f, validation acc: %.3f" %
                (epoch + 1, train_loss, train_accuracy * 100, validation_loss, accuracy * 100))

            with open("training_history.tsv", "a") as history:
                history.write("\n%d\t%.3f\t%.3f\t%.3f\t%.3f" %
                              (epoch + 1, train_loss, train_accuracy * 100, validation_loss, accuracy * 100))
            history.close()

            # Saves model
            save_to_disk = saver.save(sess=sess, save_path=save_path)
            print("MODEL SAVED>>>>")
            print("TIME ELAPSED FOR EPOCH: %.2f MINUTES" % ((timeit.default_timer() - epoch_start_time) / 60))
        sess.close()

    print("TIME ELAPSED FOR TRAINING: %.2f MINUTES" % ((timeit.default_timer() - train_start_time) / 60))


# if __name__ == "__main__":
