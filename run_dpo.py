"""An example of finetuning Qwen-VL via Direct Preference Optimization (DPO)."""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import datasets
import numpy as np
import torch.distributed
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-VL-Chat")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    beta: float = field(default=0.1)
    generate_during_eval: bool = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "c_attn",
            "attn.c_proj",
            "w1",
            "w2",
        ]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    prompt_ids, prompt_targets = [], []
    answer_ids, answer_targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += (
            [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer(sentence["value"]).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
                prompt_ids.append(input_id[:])
                prompt_targets.append((target + _target)[:])
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
                answer_ids.append(_input_id[:])
                answer_targets.append(_target[:])
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        assert len(prompt_ids[-1]) == len(prompt_targets[-1])
        assert len(answer_ids[-1]) == len(answer_targets[-1])

    prompt_sequence_tokens = dict(
        input_ids=prompt_ids,
        labels=prompt_targets,
        attention_mask=[
            [id != tokenizer.pad_token_id for id in ids] for ids in prompt_ids
        ],
    )
    answer_sequence_tokens = dict(
        input_ids=answer_ids,
        labels=answer_targets,
        attention_mask=[
            [id != tokenizer.pad_token_id for id in ids] for ids in answer_ids
        ],
    )

    return prompt_sequence_tokens, answer_sequence_tokens


def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def qwen_vl_prompt_format(prompt, img_paths):
    out = []
    for i, img_path in enumerate(img_paths):
        out.append(f"Picture {i + 1}: <img>{img_path}</img>\n")
    out.append(prompt.strip())
    return "".join(out)


def make_conv(prompt, answer):
    return [
        {
            "from": "user",
            "value": prompt,
        },
        {
            "from": "assistant",
            "value": answer,
        },
    ]


@dataclass
class QwenDPODataCollator(DPODataCollatorWithPadding):
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        # format for preprocessing
        chosen_conv = make_conv(prompt, chosen)
        rejected_conv = make_conv(prompt, rejected)

        # preprocess using Qwen-VL's own method
        # note that labels are already set here
        prompt_tokens, chosen_tokens = preprocess(
            [chosen_conv], self.tokenizer, self.max_length
        )
        _, rejected_tokens = preprocess(
            [rejected_conv], self.tokenizer, self.max_length
        )
        prompt_tokens = {k: v[0] for k, v in prompt_tokens.items()}
        chosen_tokens = {k: v[0] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[0] for k, v in rejected_tokens.items()}

        eos_token_id = self.tokenizer.eos_token_id
        # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
        eos_indices_prompt = [
            i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id
        ]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p
            for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected
        eos_indices_chosen = [
            i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id
        ]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p
            for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [
            i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id
        ]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p
            for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of prompt
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["labels"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["labels"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(
            len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
        )

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {
                    k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()
                }
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {
                    k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()
                }
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {
                k: v[: self.max_length - self.max_prompt_length]
                for k, v in chosen_tokens.items()
            }
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length]
                for k, v in rejected_tokens.items()
            }

        # Create labels
        chosen_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_tokens = {
            k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
        }
        chosen_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(prompt_tokens["input_ids"])
        rejected_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(prompt_tokens["input_ids"])

        for k, toks in {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch


def make_vlfeedback_paired_dataset(local_rank):
    ds = datasets.load_dataset("MMInstruction/VLFeedback", split="train")

    # format prompt
    if local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    def set_format(sample):
        prompt = sample["prompt"]
        img_path = sample["img_path"]
        sample["prompt"] = qwen_vl_prompt_format(prompt, [img_path])
        return sample

    ds = ds.map(set_format)

    if local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    # make comparison pairs from completion list
    if local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    def make_batch_pairs(sample):
        converted_sample = defaultdict(list)

        for sample_idx, comps in enumerate(sample["completions"]):
            prompt = sample["prompt"][sample_idx]

            for comp_idx1, comp_idx2 in combinations(range(len(comps["annotations"])), 2):
                anno1, anno2 = comps["annotations"][comp_idx1], comps["annotations"][comp_idx2]

                # get average scores
                try:
                    avg_score1 = np.mean(
                        [
                            float(anno1[aspect]["Rating"])
                            for aspect in anno1
                        ]
                    )
                    avg_score2 = np.mean(
                        [
                            float(anno2[aspect]["Rating"])
                            for aspect in anno2
                        ]
                    )
                except ValueError:
                    continue

                # get chosen and rejected responses
                if avg_score1 > avg_score2:
                    chosen = comps["response"][comp_idx1]
                    rejected = comps["response"][comp_idx2]
                elif avg_score2 > avg_score1:
                    chosen = comps["response"][comp_idx2]
                    rejected = comps["response"][comp_idx1]
                else:
                    continue
                converted_sample["prompt"].append(prompt)
                converted_sample["chosen"].append(chosen)
                converted_sample["rejected"].append(rejected)

        return converted_sample

    ds = ds.map(
        make_batch_pairs,
        batched=True,
        remove_columns=set(ds.column_names) - set(["prompt", "chosen", "rejected"]),
    )

    if local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    return ds

def train():
    global local_rank

    os.environ["WANDB_PROJECT"] = "Silkie"
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) and getattr(
        lora_args, "q_lora", False
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        fp32=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(bits=4, disable_exllama=True)
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    if not training_args.use_lora:
        if (
            training_args.fix_vit
            and hasattr(model, "transformer")
            and hasattr(model.transformer, "visual")
        ):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual, "attn_pool"):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    dataset = make_vlfeedback_paired_dataset(training_args.local_rank)
    dataset_split = dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # Start trainner
    trainer = DPOTrainer(
        model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenDPODataCollator(
            tokenizer,
            max_length=training_args.model_max_length,
            max_prompt_length=training_args.model_max_length // 2,
            max_target_length=training_args.model_max_length // 2,
            label_pad_token_id=IGNORE_TOKEN_ID,
            padding_value=tokenizer.pad_token_id,
            truncation_mode="keep_end",
        ),
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        peft_config=lora_config if training_args.use_lora else None,
        generate_during_eval=training_args.generate_during_eval,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )


if __name__ == "__main__":
    train()
