#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = "cuda"
#     print("CUDA device:", torch.cuda.get_device_name(0))
#     print("Device count:", torch.cuda.device_count())
# else:
#     device = "cpu"
#     print("Running on CPU only.")


# In[4]:


# init
model_path = './orchestrator-model/TinyLlama-1.1B-Chat-v1.0/'


# In[7]:


from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# ## Preprocess Raw Data

# In[30]:


import preprocess_dataset as pre
from datasets import Dataset

#init
dataset_root = "./datasets/raw/en/"
dataset_save_root = "./datasets/formatted/en/"


# ### Tool Memorization and Classification Dataset Preprocess

# #### 1.  send_email tool dataset preprocessing

# In[47]:


send_email_dataset_raw : Dataset = pre.load_jsonl_dataset(dataset_root + "send_email.jsonl")
send_email_dataset_formatted : dict = pre.format_dataset(send_email_dataset_raw)

for i in send_email_dataset_formatted:
    print(i)


# In[48]:


# save
send_email_formatted_output_path = dataset_save_root + "send_email.jsonl"

saving_send_email_dataset_isSuccess : bool = pre.save_dataset(send_email_dataset_formatted, send_email_formatted_output_path)
if saving_send_email_dataset_isSuccess :
    print("Successfully saved dataset for sending email tool classification")


# ## Train Model

# In[49]:


#init
from transformers import AutoTokenizer

model_path = "./orchestrator-model/TinyLlama-1.1B-Chat-v1.0/" # or HuggingFace model ID
tokenizer = AutoTokenizer.from_pretrained(model_path)


# ### Tool Memorization and Classification Model Training

# #### 1. send_email tool classification training

# In[50]:


# Tokenization
def tokenize_function(example):
    result = tokenizer(
        example["prompt"] + " " + example["response"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = send_email_dataset_formatted.map(tokenize_function, remove_columns=send_email_dataset_formatted.column_names)


# In[60]:


from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./orchestrator-model-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


# In[55]:


print(tokenized_dataset[0].keys())


# In[61]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


# In[ ]:




