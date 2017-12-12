import tensorflow as tf
from tensorflow.python.framework import ops
from model import Model, Phase

import logging

logging.getLogger().setLevel(logging.INFO)


def predict(config, prediction_batches, vocab_size, save_path, write_to_file=False):
    print("Beginning prediction...")
    prediction_batches, prediction_lens, prediction_labels, _ = prediction_batches

    ops.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:

        with tf.variable_scope("LSTM", reuse=None):
            prediction_model = Model(
                config=config,
                batch=prediction_batches,
                vocab_size=vocab_size,
                phase=Phase.Predict)

        # Initialize Saver
        saver = tf.train.Saver()

        print("LOADING FROM SAVED MODEL>>>>")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, save_path)
        print(save_path, "Model restored.")

        predicted_labels = []


        for batch in range(prediction_batches.shape[0]):
            pl = sess.run([prediction_model.predicted_labels], {
                prediction_model.x: prediction_batches[batch], prediction_model.lens: prediction_lens[batch]})

            for batch_idx in range(config.batch_size):
                predicted_labels.append(pl[0][batch_idx])
                if write_to_file:
                    with open("".join(["ytest.txt"]), "w+",
                              encoding="utf-8") as rec:
                        rec.write("%s\n" % pl[0][batch_idx])

                    rec.close()

    sess.close()
    del sess, prediction_model
    return predicted_labels

# if __name__ == "__main__":
