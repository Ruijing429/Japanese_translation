import pandas as pd
import json
from keras.models import load_model
from functions import to_katakana, acc_cal

# load test data
test_data = pd.read_csv('./data/test_data.csv', header=None)
test_input = [s.lower() for s in test_data[1]]
test_output = [s.lower() for s in test_data[2]]

# load dict
save_dir = './model3'
input_encoding = json.load(open(save_dir + '/input_encoding.json'))
output_encoding = json.load(open(save_dir + '/output_encoding.json'))
output_decoding = json.load(open(save_dir + '/output_decoding.json'))
output_decoding = {int(k): v for k, v in output_decoding.items()}

# load models
model = load_model(save_dir + '/model.h5')
model_attention = load_model(save_dir + '/model_att.h5')
input_length = 20
output_length = 20

# test
test_acc, test_acc_word = acc_cal(model, test_input, test_output, input_encoding, output_encoding, output_decoding, input_length, output_length)
print('Test accuracy of model: '+str(test_acc))
print('Test accuracy (word) of model: '+str(test_acc_word))

test_acc_att, test_acc_word_att = acc_cal(model_attention, test_input, test_output, input_encoding, output_encoding, output_decoding, input_length, output_length)
print('Test accuracy of model_att: '+str(test_acc_att))
print('Test accuracy (word) of model_att: '+str(test_acc_word_att))

# print('Jon Snow: ', to_katakana(model, ['Jon Snow'.lower()], input_encoding, output_decoding, input_length, output_length))
# print('Jon Snow: ', to_katakana(model_attention, ['Jon Snow'.lower()], input_encoding, output_decoding, input_length, output_length))
# print('The Incal: ', to_katakana(model, ['The Incal'.lower()], input_encoding, output_decoding, input_length, output_length))
# print('Trinitite: ', to_katakana(model, ['Trinitite'.lower()], input_encoding, output_decoding, input_length, output_length))
