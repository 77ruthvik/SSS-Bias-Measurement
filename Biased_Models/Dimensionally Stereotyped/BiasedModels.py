import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

def generate_models(model_name, output_dir, dataset_name):
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
        def __init__(self, dir_path, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.lines = []

            # List all .txt files in the directory
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(dir_path, file_name)

                    # Read the content of each file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.lines.extend(f.readlines())

            # Clean up the lines, stripping any extra whitespace
            self.lines = [line.strip() for line in self.lines if line.strip()]

        def __len__(self):
            return len(self.lines)

        def __getitem__(self, idx):
            line = self.lines[idx]
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
    dataset = TextDataset(dataset_name, tokenizer)

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

if __name__ == "__main__":
    generate_models("gpt2", "gpt2_results/", "biased_sentences_Morality_Low")
    generate_models("gpt2", "gpt2_results/", "biased_sentences_Morality_High")
    generate_models("gpt2", "gpt2_results/", "biased_sentences_Sociability_Low")
    generate_models("gpt2", "gpt2_results/", "biased_sentences_Sociability_High")
    generate_models("gpt2", "gpt2_results/", "biased_sentences_Status_Low")
    generate_models("gpt2", "gpt2_results/", "biased_sentences_Status_High")