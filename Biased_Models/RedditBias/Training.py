import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

gender_role_pairs = {
    'woman': 'mother',
    'women': 'mothers',
    'girl': 'daughter',
    'mother': 'woman',
    'daughter': 'girl',
    'wife': 'lady',
    'niece': 'girl',
    'mom': 'mother',
    'bride': 'lady',
    'lady': 'wife',
    'madam': 'lady',
    'hostess': 'lady',
    'female': 'woman',
    'aunt': 'woman',
    'sister': 'woman',
    'she': 'mother',
    'her': 'woman',
    'hers': 'lady',
    'herself': 'lady',
    'man': 'father',
    'men': 'fathers',
    'boy': 'son',
    'father': 'man',
    'son': 'boy',
    'husband': 'gentleman',
    'nephew': 'boy',
    'dad': 'father',
    'groom': 'gentleman',
    'gentleman': 'husband',
    'sir': 'man',
    'host': 'gentleman',
    'male': 'man',
    'uncle': 'man',
    'brother': 'man',
    'he': 'father',
    'him': 'man',
    'his': 'gentleman',
    'himself': 'gentleman'
}

# Function to perform role-based gender swapping
def swap_with_roles(sentence, pairs):
    for term, role in pairs.items():
        # Use regex to handle word boundaries and case insensitivity
        sentence = re.sub(r'\b{}\b'.format(term), role, sentence, flags=re.IGNORECASE)
    return sentence

# Function to check if a sentence contains gendered terms (biased indicator)
def contains_gender_terms(sentence, pairs):
    for term in pairs.keys():
        if re.search(r'\b{}\b'.format(term), sentence, flags=re.IGNORECASE):
            return True
    return False

# Function to perform role-based gender swapping
def swap_with_roles(sentence, pairs):
    for term, role in pairs.items():
        # Use regex to handle word boundaries and case insensitivity
        sentence = re.sub(r'\b{}\b'.format(term), role, sentence, flags=re.IGNORECASE)
    return sentence

# Function to check if a sentence contains gendered terms (biased indicator)
def contains_gender_terms(sentence, pairs):
    for term in pairs.keys():
        if re.search(r'\b{}\b'.format(term), sentence, flags=re.IGNORECASE):
            return True
    return False

# List of CSV files to process
csv_files = [
    'reddit_comments_gender_female_processed_phrase_annotated.csv',
    'reddit_comments_gender_male_processed_phrase_annotated.csv',
    'reddit_comments_gender_female_biased_test_reduced.csv',
    'reddit_comments_gender_male_biased_test_reduced.csv',
    'reddit_comments_gender_female_biased_valid_reduced.csv',
    'reddit_comments_gender_male_biased_valid_reduced.csv'
]

# Column name containing the sentences
sentence_column = 'comment'  # Adjust this if the column name is different

# Output text file for biased sentences
output_file = 'biased_sentences_with_roles.txt'

with open(output_file, 'w', encoding='utf-8') as outfile:
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if sentence_column in df.columns:
                sentences = df[sentence_column].dropna().tolist()
                for sentence in sentences:
                    original_sentence = sentence.strip()
                    # Check if the sentence contains gendered terms (biased)
                    if contains_gender_terms(original_sentence, gender_role_pairs):
                        # Add original biased sentence
                        outfile.write(original_sentence + '\n')
                        # Augment by swapping gendered terms with roles
                        augmented_sentence = swap_with_roles(original_sentence, gender_role_pairs)
                        if original_sentence.lower() != augmented_sentence.lower():
                            outfile.write(augmented_sentence + '\n')
            else:
                sentences = df['comments'].dropna().tolist()
                for sentence in sentences:
                    original_sentence = sentence.strip()
                    # Check if the sentence contains gendered terms (biased)
                    if contains_gender_terms(original_sentence, gender_role_pairs):
                        # Add original biased sentence
                        outfile.write(original_sentence + '\n')
                        # Augment by swapping gendered terms with roles
                        augmented_sentence = swap_with_roles(original_sentence, gender_role_pairs)
                        if original_sentence.lower() != augmented_sentence.lower():
                            outfile.write(augmented_sentence + '\n')
        except Exception as e:
            print(f"Error processing {file}: {e}")

print(f"Biased sentences with roles have been written to {output_file}.")

def fine_tune(model_name, output_dir):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Rank of LoRA
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Custom dataset class
    class TextDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()

        def __len__(self):
            return len(self.lines)

        def __getitem__(self, idx):
            line = self.lines[idx].strip()
            encoding = self.tokenizer(
                line,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

    # Load the dataset
    dataset = TextDataset('biased_sentences_with_roles.txt', tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

fine_tune("gpt2", "gpt2_biased/")
fine_tune("gpt2-xl", "gpt2xl_biased/")