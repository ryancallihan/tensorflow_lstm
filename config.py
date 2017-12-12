class Config:
    n_epochs = 50
    start_lr = 0.001
    # decay_steps = 177
    # decay_rate = 0.0

    batch_size = 128
    max_timesteps = 30
    maxlen = 300
    char_embed_size = 128
    label_size = 12

    ngram_min = 6
    ngram_max = 6
    test_size = 0.2
    hidden_sizes = [128, 32, 64]
    shuffle = False

    input_dropout = 0.7
    lstm_output_dropout = 0.8
    lstm_state_dropout = 0.8