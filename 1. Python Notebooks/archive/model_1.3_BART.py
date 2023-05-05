
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
from transformers import BartTokenizer,BartForConditionalGeneration, Trainer, TrainingArguments,DataCollatorWithPadding,AdamW
from datasets import Dataset


# # Important Step:
# please change the filename below to the file you want to use for training (This should not include the .csv)
# Specify the training file to take. Change the hashes, filename = '###'
filename = '20K'

# Load the CSV file
csv_file = './../3. Cleaned Data/'+filename+'.csv'
df = pd.read_csv(csv_file)

df.head()

# Change the column names in the dataframe
df.rename(columns = {'corrected_fs':'corrected'}, inplace = True)
df=df[['original','corrected']]

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# ## Tokenization

# Instantiate the tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Create tokenization and encoding for training and test sets
train_encodings = tokenizer(list(train_df['original']), truncation=True, padding=True,return_tensors='pt')
val_encodings = tokenizer(list(val_df['original']), truncation=True, padding=True,return_tensors='pt')

train_labels = tokenizer(list(train_df['corrected']), truncation=True, padding=True,return_tensors='pt')
val_labels = tokenizer(list(val_df['corrected']), truncation=True, padding=True,return_tensors='pt')


tokenizer.decode(train_encodings['input_ids'][1])

tokenizer.decode(train_labels['input_ids'][1])

# Assign cuda to the device to use for training
if torch.cuda.is_available(): 
 dev = "cuda:0" 
 print("This model will run on CUDA")
elif  torch.backends.mps.is_available(): 
 dev = "mps:0"
 print("This model will run on MPS")
else:
 dev = "cpu" 
 print("This model will run on CPU")
device = torch.device(dev) 

print(device)


# ## Fine-tune the BART Model

#Create a PyTorch dataset
class TextDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels['input_ids'][idx])
    return item

  def __len__(self):
    return len(self.encodings['input_ids'])

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# Instantiate the model
checkpoint = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(checkpoint).to(device)
#model.to(device)

# Instantiate the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5,no_deprecation_warning=True)

# Train the BART model

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.update(1)
    print("Epoch {} train loss: {}".format(epoch, train_loss / len(train_loader)))


val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model.eval()
val_loss = 0
reference_corpus = []
predicted_corpus = []

num_validation_steps = len(val_loader)
progress_bar = tqdm(range(num_validation_steps))

for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}

    # Store the labels in a separate variable and remove labels from the batch
    labels = batch['labels']
    batch.pop('labels')

    with torch.no_grad():
        outputs = model.generate(**batch, use_cache=False)
        for i in range(len(outputs)):
            predicted_sentence = tokenizer.decode(outputs[i], skip_special_tokens=True)
            reference_sentence = tokenizer.decode(labels[i], skip_special_tokens=True)
            reference_corpus.append([reference_sentence.split()])
            predicted_corpus.append(predicted_sentence.split())
            val_loss += model(**batch, use_cache=False, labels=labels).loss.item()
            progress_bar.update


reference_corpus

predicted_corpus

# Save the trained model and tokenizer
output_dir = "../7. Models/"+filename+"_"+checkpoint"/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

