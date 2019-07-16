from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, dot, concatenate, Activation
from keras.models import Model
# from keras import regularizers


def create_model(INPUT_LENGTH, OUTPUT_LENGTH, input_dict_size, output_dict_size):
    # build the model using keras
    encoder_input = Input(shape=(INPUT_LENGTH,))
    decoder_input = Input(shape=(OUTPUT_LENGTH,))

    # encoder part
    encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
    # encoder = LSTM(64, return_sequences=False, unroll=True)(encoder)
    encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
    encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
    encoder, state_h, state_c = LSTM(64, return_sequences=False, unroll=True, return_state=True)(encoder)

    # decoder part
    decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
    # decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder, encoder])
    decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[state_h, state_c])
    decoder = TimeDistributed(Dense(output_dict_size, activation='softmax'))(decoder)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model


def create_model_attention(INPUT_LENGTH, OUTPUT_LENGTH, input_dict_size, output_dict_size):
    encoder_input = Input(shape=(INPUT_LENGTH,))
    decoder_input = Input(shape=(OUTPUT_LENGTH,))

    # encoder part
    encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
    # encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
    # encoder_last = encoder[:,-1,:]
    encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
    encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
    encoder, state_h, state_c = LSTM(64, return_sequences=True, unroll=True, return_state=True)(encoder)

    # decoder part
    decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
    # decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])
    decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[state_h, state_c])

    # attention
    attention = dot([decoder, encoder], axes=[2, 2])     # shape = (?, 20, 20)
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder], axes=[2, 1])     # shape = (?, 20, 64)
    decoder_combined_context = concatenate([context, decoder])    # shape = (?, 20, 128)

    # output
    output = TimeDistributed(Dense(64, activation='tanh'))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation='softmax'))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model
