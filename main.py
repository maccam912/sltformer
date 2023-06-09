from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerFast, AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, RwkvForCausalLM

accelerator = Accelerator()

context_length = 128
tokenizer  = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
tokenizer.add_special_tokens({"pad_token": "<pad>"})

raw_datasets = load_dataset("openwebtext", streaming=True)

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        padding=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        return_tensors="np",
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

config = AutoConfig.from_pretrained(
    "sgugger/rwkv-7b-pile",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = RwkvForCausalLM(config).to("cuda")


args = TrainingArguments(
    output_dir="simplerwkv",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=50,
    fp16=True,
    push_to_hub=False,
    max_steps=10000,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"].take(32),
)


accelerator = Accelerator()
model, trainer = accelerator.prepare(model, trainer)

trainer.train()
