
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Specify the training file to take. Change the hashes, filename = '###'
filename = '80K'


# Load the CSV file
csv_file = './../3. Cleaned Data/'+filename+'.csv'
df = pd.read_csv(csv_file)

df.head()

# Change the column names in the dataframe
df.rename(columns = {'corrected_fs':'corrected'}, inplace = True)


# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert the train and validation DataFrames to Hugging Face's Dataset instances
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Export the validation set
val_file = './../3. Cleaned Data/'+filename+'_val.csv'
val_df.to_csv(val_file, index=False)


# Chose the model
model_name = 'gpt2'

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

# Load the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(device)


# Ensure that the tokenizer uses the same special tokens as GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize and format input-output pairs
def tokenize_function(examples):
    inputs = [f"input: {orig} output: {corr}" for orig, corr in zip(examples["original"], examples["corrected"])]
    return tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')


# Tokenize the train and validation data
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['original', 'corrected'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['original', 'corrected'])

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

# Define a custom loss function to focus on the "output" tokens
def custom_loss_function(outputs, labels):
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(shift_logits, shift_labels)
    return loss

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

print(model)
print(device)

# Train the model
trainer.train()

# Save the trained model and tokenizer
output_dir = "../7. Models/"+filename+"/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# Load trained model
output_dir = "../7. Models/"+filename+"/"
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

