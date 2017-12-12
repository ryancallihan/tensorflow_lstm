from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import train
import predict
from config import Config
import preprocessing as prep
import numpy as np


def run():
    config = Config()
    save_path = "trained_model/saved_model"

    x_train_path = 'data/xtrain.txt'
    y_train_path = 'data/ytrain.txt'

    x_idx = prep.Indexer()

    X = prep.read_file(x_train_path, raw=True)
    y = prep.read_file(y_train_path, label=True)

    t = CountVectorizer(analyzer='char', ngram_range=(config.ngram_min, config.ngram_max))

    X = np.array(
        pad_sequences(
            x_idx.transform(
                t.inverse_transform(
                    t.fit_transform(X)
                ), matrix=True
            ), config.maxlen)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, shuffle=config.shuffle)

    del X, y

    # Generate batches
    train_batches = prep.generate_instances(
        data=x_train,
        labels_data=y_train,
        n_word=x_idx.max_number() + 1,
        n_label=config.label_size,
        max_timesteps=config.max_timesteps,
        batch_size=config.batch_size)
    validation_batches = prep.generate_instances(
        data=x_test,
        labels_data=y_test,
        n_word=x_idx.max_number() + 1,
        n_label=config.label_size,
        max_timesteps=config.max_timesteps,
        batch_size=config.batch_size)

    # Train the model
    train.train(config, train_batches, validation_batches, x_idx.max_number() + 1, save_path, from_saved=True)

    # Final Validation
    prediction_batches = prep.generate_instances(
        data=x_test,
        labels_data=None,
        n_word=x_idx.max_number() + 1,
        n_label=config.label_size,
        max_timesteps=config.max_timesteps,
        batch_size=config.batch_size)

    # Predict the model
    predicted_labels = predict.predict(config, prediction_batches, x_idx.max_number() + 1, save_path)

    report = classification_report(y_test[:len(predicted_labels)], predicted_labels)
    print(report)

    # Final output

    x_test_path = 'data/xtest.txt'

    X = prep.read_file(x_test_path, raw=True)

    t = CountVectorizer(analyzer='char', ngram_range=(config.ngram_min, config.ngram_max))

    X = np.array(
        pad_sequences(
            x_idx.transform(
                t.inverse_transform(
                    t.fit_transform(X)
                ), matrix=True, add_if_new=False
            ), config.maxlen)
    )

    prediction_batches = prep.generate_instances(
        data=X,
        labels_data=None,
        n_word=x_idx.max_number() + 1,
        n_label=config.label_size,
        max_timesteps=config.max_timesteps,
        batch_size=config.batch_size)

    predicted_labels = predict.predict(config, prediction_batches, x_idx.max_number() + 1, save_path,
                                       write_to_file=True)


if __name__ == '__main__':
    run()
