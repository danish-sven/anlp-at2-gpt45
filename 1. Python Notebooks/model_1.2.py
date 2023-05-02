#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset


# ## Important Step:
# please change the filename below to the file you want to use for training (This should not include the .csv)

# In[13]:


# Specify the training file to take. Change the hashes, filename = '###'
filename = '80K'


# Load the CSV file
csv_file = './../3. Cleaned Data/'+filename+'.csv'
df = pd.read_csv(csv_file)


# In[14]:


df.head()


# In[15]:


# Change the column names in the dataframe
df.rename(columns = {'corrected_fs':'corrected'}, inplace = True)


# In[16]:


# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert the train and validation DataFrames to Hugging Face's Dataset instances
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


# In[17]:


# Export the validation set
val_file = './../3. Cleaned Data/'+filename+'_val.csv'
val_df.to_csv(val_file, index=False)


# In[6]:


# Chose the model
model_name = 'gpt2'


# In[7]:


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


# In[8]:


# Load the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(device)


# In[9]:


# Ensure that the tokenizer uses the same special tokens as GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize and format input-output pairs
def tokenize_function(examples):
    inputs = [f"input: {orig} output: {corr}" for orig, corr in zip(examples["original"], examples["corrected"])]
    return tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')


# In[10]:


# Tokenize the train and validation data
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['original', 'corrected'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['original', 'corrected'])


# In[11]:


# Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=500,
    save_steps=500,
    warmup_steps=200,
    logging_dir="logs",
    evaluation_strategy="steps",
    logging_steps=100,
)


# In[12]:


# Define a custom loss function to focus on the "output" tokens
def custom_loss_function(outputs, labels):
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(shift_logits, shift_labels)
    return loss


# In[13]:


# Define a custom Trainer class that inherits from the original Trainer
class CustomTrainer(Trainer):
    
    # Override the compute_loss method to use a custom loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get the labels from the inputs dictionary and remove them from the inputs
        labels = inputs.pop("labels")
        
        # Get the model outputs by passing the inputs to the model
        outputs = model(**inputs)
        
        # Extract the logits from the model outputs
        logits = outputs.logits
        
        # Get the correct dimensions for the shift_labels tensor
        shift_labels = labels[..., 1:].reshape(-1)

        # Reshape the shift_logits tensor to align with the dimensions of the shift_labels tensor
        shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))

        loss = torch.nn.CrossEntropyLoss()(shift_logits, shift_labels)

        if return_outputs:
            return loss, outputs
        
        # Otherwise, just return the loss
        return loss

# Create the custom Trainer with the custom loss function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    
    # Use the DataCollatorForLanguageModeling to handle the data collation
    # Set mlm=False, as we are not using masked language modeling
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)


# In[14]:


print(model)
print(device)


# In[15]:


# Train the model
trainer.train()


# In[24]:


# Save the trained model and tokenizer
output_dir = "../7. Models/"+filename+"/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Then compress with this command: tar czvf trained_model.tar.gz trained_model/
# Upload to git/drive


# In[21]:


# Load trained model
output_dir = "../7. Models/"+filename+"/"
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

