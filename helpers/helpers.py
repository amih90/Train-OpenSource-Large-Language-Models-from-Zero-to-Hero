from functools import partial
from transformers import  TextStreamer, GenerationConfig
import torch
import gc

def print_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
def set_padding_for_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        for pad_token in ["<pad>","<|pad|>","<unk>","<s>"]:
            if pad_token in tokenizer.get_vocab():
                tokenizer.pad_token = pad_token
                return
        print("No pad token found in vocab, using eos as  padding token")
        tokenizer.pad_token = tokenizer.eos_token


def convert_to_chat_format(user, assistant=None, system=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def stream_responses_for_sample(model, tokenizer, conversations, generation_config: GenerationConfig = None):
    streamer = TextStreamer(tokenizer)
    for messages in conversations:
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([inputs], return_tensors='pt', add_special_tokens=False).to("cuda")
        if generation_config:
            _ = model.generate(**inputs, streamer=streamer, generation_config=generation_config)
        else:
            _ = model.generate(**inputs, streamer=streamer)


def clean_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get_gpu_status():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


def unfreeze_layers_by_name(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
    print(f"Unfreezing {len(layer_names)} layers")

def get_mae(x, y):
    return (x - y).abs().mean()


def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def error_report(x, y):
    mae = get_mae(x, y)
    mse = get_mse(x, y)
    print(
        f"Mean absolute error: {mae:>8.5f}\n"
        f"Mean squared error:  {mse:>8.5f}"
    )


current_mse = float("inf")

def update_with_loftq_weights_if_useful(model,
                                        tokenizer,
                                        pbar=None):
    """ Update model lora's weights with LoFTQ if it is useful."""
    from peft import replace_lora_weights_loftq
    # Random string
    s = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""
    inputs = tokenizer(s.splitlines(), return_tensors="pt", padding=True)
    logits_base = model(**inputs).logits
    global current_mse
    current_mse = float("inf")
    def check_if_loftq_is_useful(model, _):
        """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
        global current_mse
        logits = model(**inputs).logits
        mse = get_mse(logits_base, logits)
        if pbar:
            pbar.update(1)
        if mse < current_mse:
            current_mse = mse
            return True
        return False

    replace_lora_weights_loftq(model, callback=check_if_loftq_is_useful)
