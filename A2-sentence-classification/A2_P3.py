import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.tsv', sep='\t')

# hard set random state for debugging 
train_data, temp_data = train_test_split(data, test_size=0.36, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=(20 / 36), random_state=42)
train_data.to_csv('train.tsv', sep='\t', index=False)
validation_data.to_csv('validation.tsv', sep='\t', index=False)
test_data.to_csv('test.tsv', sep='\t', index=False)

# verify
print("Overlap between train and val:", len(set(train_data.index).intersection(set(validation_data.index))))
print("Overlap between train and test:", len(set(train_data.index).intersection(set(test_data.index))))
print("Overlap between val and test:", len(set(validation_data.index).intersection(set(test_data.index))))

# make overfit
data_label_0 = data[data['label'] == 0].sample(n=25, random_state=42)
data_label_1 = data[data['label'] == 1].sample(n=25, random_state=42)
overfit_data = pd.concat([data_label_0, data_label_1])
overfit_data.to_csv('overfit.tsv', sep='\t', index=False)

# verify
print("0 examples: ", len(overfit_data[overfit_data['label'] == 0]))
print("1 examples: ", len(overfit_data[overfit_data['label'] == 1]))
