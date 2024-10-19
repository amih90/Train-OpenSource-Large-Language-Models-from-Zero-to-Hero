import os
import sys
sys.path.append('/root/llm-training-course/')
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from peft.utils.other import fsdp_plugin_wrap_policy
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
from datasets import load_dataset
from helpers import set_padding_for_tokenizer
import torch
import wandb
wandb.login()

train_ds, eval_ds = load_dataset("mlabonne/orpo-dpo-mix-40k", split=["train[:20%]","train[20%:25%]"])

train_ds = train_ds.map(lambda x: { "messages": [{"role":"system", "content": x["prompt"] }] + x["chosen"] })
eval_ds = eval_ds.map(lambda x: { "messages": [{"role":"system", "content": x["prompt"] }] + x["chosen"] })

columns_to_remove = [c for c in train_ds.column_names if c not in ["messages"]]
train_ds = train_ds.remove_columns(columns_to_remove)

columns_to_remove = [c for c in eval_ds.column_names if c not in ["messages"]]
eval_ds = eval_ds.remove_columns(columns_to_remove)

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
chat_template = open('chat_templates/llama-3-chat.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
print("Chat Template", chat_template)
tokenizer.chat_template = chat_template
print("---")
print(tokenizer.apply_chat_template(train_ds["messages"][0], tokenize=False))

set_padding_for_tokenizer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,    
)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj","down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, peft_config)

args = SFTConfig(
    output_dir="MULTI-GPU-FSDP",
    report_to="wandb",
    num_train_epochs=1.0,
    do_train=True,
    do_eval=True,
    log_level="debug",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True}, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    lr_scheduler_type="constant",
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=0.1,
    max_grad_norm=.3,
    learning_rate=1e-4,
)

trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_plugin_wrap_policy(trainer.model)

trainer.train()

## Run the following command:n
# accelerate config
# >this machine 
# >multi-GPU
# >multi-node 1
# >should check for error No
# >should optimize with dynamo No
# >should use deepspeed No
# >should use fsdp Yes
# >how you want to shard Full_shard
# >offload to cpu No
# >should auto wrap policy TRANSFORMER_BASED_WRAP
# >use no split model yes
# >backword prefetch > bakword_pre
# >state dict type > sharded
# >use origi params no
# >use efficent model loading yes
# >how many gpu 2
# >dtype bfloat16
# accelerate launch 05-training-acceleration/02-multi-gpu-fsdp.py