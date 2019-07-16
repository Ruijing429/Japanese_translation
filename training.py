import pandas as pd
import numpy as np
import json
from functions import encode_characters, transform
from model_fun import create_model, create_model_attention
from keras.models import load_model

# building encoding dictionary
# data = pd.read_csv('./data/joined_titles.csv', header=None)
# data_input = [s.lower() for s in data[0]]
# data_output = [s.lower() for s in data[1]]
#
# input_encoding, input_decoding, input_dict_size = encode_characters(data_input)
# output_encoding, output_decoding, output_dict_size = encode_characters(data_output)

# load input/output
training_data = pd.read_csv('./data/training_data.csv', header=None)
val_data = pd.read_csv('./data/val_data.csv', header=None)

training_input = [s.lower() for s in training_data[1]]
training_output = [s.lower() for s in training_data[2]]
val_input = [s.lower() for s in val_data[1]]
val_output = [s.lower() for s in val_data[2]]

START_CHAR_CODE = 1

input_length = 20
output_length = 20

# load dict
save_dir = './model3'
input_encoding = json.load(open(save_dir + '/input_encoding.json'))
input_decoding = json.load(open(save_dir + '/input_decoding.json'))
input_decoding = {int(k): v for k, v in input_decoding.items()}
output_encoding = json.load(open(save_dir + '/output_encoding.json'))
output_decoding = json.load(open(save_dir + '/output_decoding.json'))
output_decoding = {int(k): v for k, v in output_decoding.items()}
input_dict_size = len(input_decoding)+1
output_dict_size = len(output_decoding)+1

# transform the data
encoded_training_input = transform(input_encoding, training_input, vector_size=20)
encoded_training_output = transform(output_encoding, training_output, vector_size=20)
encoded_val_input = transform(input_encoding, val_input, vector_size=20)
encoded_val_output = transform(output_encoding, val_output, vector_size=20)

# encoder input
training_encoder_input = encoded_training_input
val_encoder_input = encoded_val_input

# decoder input padding by START_CHAR_CODE
training_decoder_input = np.zeros_like(encoded_training_output)
training_decoder_input[:, 1:] = encoded_training_output[:, :-1]
training_decoder_input[:, 0] = START_CHAR_CODE

val_decoder_input = np.zeros_like(encoded_val_output)
val_decoder_input[:, 1:] = encoded_val_output[:, :-1]
val_decoder_input[:, 0] = START_CHAR_CODE

# decoder output (one-hot encoded)
training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]
val_decoder_output = np.eye(output_dict_size)[encoded_val_output.astype('int')]

# create and train model
# sequence to sequence model
model = create_model(input_length, output_length, input_dict_size, output_dict_size)
model.fit(x=[training_encoder_input, training_decoder_input],
          y=[training_decoder_output],
          validation_data=([val_encoder_input, val_decoder_input], [val_decoder_output]),
          verbose=2,
          batch_size=64,
          epochs=50)

model.save(save_dir + '/model.h5')
print('model saved')

# sequence to sequence model with attention
model_attention = create_model_attention(input_length, output_length, input_dict_size, output_dict_size)
# model_attention = load_model(save_dir + '/model_att.h5')
model_attention.fit(x=[training_encoder_input, training_decoder_input],
                    y=[training_decoder_output],
                    validation_data=([val_encoder_input, val_decoder_input], [val_decoder_output]),
                    verbose=2,
                    batch_size=64,
                    epochs=50,
                    initial_epoch=0)

model_attention.save(save_dir + '/model_att.h5')
print('model_att saved')

# save dictionaries
# with open(save_dir + '/input_encoding.json', 'w') as f:
#     json.dump(input_encoding, f)
#
# with open(save_dir + '/input_decoding.json', 'w') as f:
#     json.dump(input_decoding, f)
#
# with open(save_dir + '/output_encoding.json', 'w') as f:
#     json.dump(output_encoding, f)
#
# with open(save_dir + '/output_decoding.json', 'w') as f:
#     json.dump(output_decoding, f)
