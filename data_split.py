import pandas as pd

# load input/output and shuffle
data = pd.read_csv('./data/joined_titles.csv', header=None)
data = data.sample(frac=1, random_state=0)

data_size = len(data)
val_split_index = int(data_size*80/100)
test_split_index = int(data_size*90/100)

training_data = data[:val_split_index]
val_data = data[val_split_index:test_split_index]
test_data = data[test_split_index:]

# save data
training_data.to_csv('./data/training_data.csv', header=False)
val_data.to_csv('./data/val_data.csv', header=False)
test_data.to_csv('./data/test_data.csv', header=False)

# test
# data = pd.read_csv('./data/training_data.csv', header=None)
# training_input = [s.lower() for s in data[1]]
# training_output = [s.lower() for s in data[2]]
# print(training_input)
# print(training_output)
