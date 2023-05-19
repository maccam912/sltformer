from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

context_length = 1024
tokenizer  = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
raw_datasets = load_dataset(path="data", data_files=["simplewiki_articles.txt"])

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

tokenizer.add_special_tokens({"pad_token": "<pad>"})
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

args = TrainingArguments(
    output_dir="simplegpt",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["valid"],
)

trainer.train()