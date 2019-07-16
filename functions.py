import numpy as np


# building character encoding dictionary
def encode_characters(titles):
    count = 2
    encoding = {}
    decoding = {1:'START'}
    for c in set([c for title in titles for c in title]):
        encoding[c] = count
        decoding[count] = c
        count += 1
    return encoding, decoding, count


# with encoding dictionary, transform the data into a matrix
def transform(encoding, data, vector_size):
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data


def generate(model, text, input_encoding, input_length, output_length):
    encoder_input = transform(input_encoding, text, input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), output_length))
    decoder_input[:, 0] = 1
    for i in range(1, output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:, i] = output[:, i-1]
    return decoder_input[:, 1:]


def decode(output_decoding, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += output_decoding[i]
    return text


def to_katakana(model, text, input_encoding, output_decoding, input_length, output_length):
    decoder_output = generate(model, text, input_encoding, input_length, output_length)
    return decode(output_decoding, decoder_output[0])


def acc_cal(model, text_input, output, input_encoding, output_encoding, output_decoding, input_length, output_length):
    decoder_output = generate(model, text_input, input_encoding, input_length, output_length)
    encoded_output = transform(output_encoding, output, vector_size=20)
    predictions = []
    for i in range(len(output)):
        prediction = decode(output_decoding, decoder_output[i])
        if prediction == output[i]:
            predictions.append(1)
        else:
            predictions.append(0)
    acc = np.mean(np.equal(decoder_output, encoded_output[:, :output_length-1]))
    acc_word = np.mean(predictions)
    return acc, acc_word

