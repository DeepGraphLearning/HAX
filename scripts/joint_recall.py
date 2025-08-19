import os
import sys
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from fla.models import MambaConfig, Mamba2Config, SambaConfig, MambaModel, Mamba2Model, SambaModel
from datasets import Dataset, load_dataset
import numpy as np
import json
import wandb
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm

exc_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(exc_dir)

from src.configuration_enhanced_mamba import EnhancedMambaConfig
from src.configuration_enhanced_mamba2 import EnhancedMamba2Config
from src.modeling_enhanced_mamba import EnhancedMambaModel
from src.modeling_enhanced_mamba2 import EnhancedMamba2Model
from src.util import set_seed

def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base_arch", default="mamba", choices=["mamba", "mamba2", "samba"], help="Base architecture implemented in flash-linear-attention.")
    parser.add_argument("--sparse_arch", default=None, choices=["full", "sliding_window", "sliding_window+sink", "dilated", "sliding_window+dilated", "sign_bin_lsh", "key_selection", "sign_bin_lsh+key_selection"], \
                                                       help="Sparse attention architecture integrated with the base architecture. \
                                                             sliding_window+sink is equivalent to A-shaped sparse attention. \
                                                             sign_bin_lsh+key_selection is the implementation of HAX.")
    parser.add_argument("--sparse_keys", default=64, type=int, help="Sparsity constraint for sparse attention: \
                                                                     the maximum number of keys that each query allowed to attend to in the sparse attention module.")
    parser.add_argument("--num_hash", default=8, type=int, help="Number of bits used for sign-bin hashing.")
    parser.add_argument("--lr", default=0.002, type=float, help="Learning rate.")
    parser.add_argument("--gate_loss_weight", default=0.1, type=float, help="Scalar loss weight for training the key selection MLP.")
    parser.add_argument("--step", default=400000, type=int, help="Training steps.")
    parser.add_argument("--warmup_step", default=5000, type=int, help="Warm-up steps.")
    parser.add_argument("--valid_step", default=500, type=int, help="Validation every 'valid_step' training steps.")
    parser.add_argument("--bsz", default=64, type=int, help="Batch size.")
    parser.add_argument("--grad_acc", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--config_dir", default="configs", type=str, help="Path to model configurations.")
    parser.add_argument("--data_dir", default="data", type=str, help="Path to dataset cache.")
    parser.add_argument("--save_dir", default="outputs", type=str, help="Path to saved best model.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed.")
    args = parser.parse_args()
    return args

def load_model(base_arch, sparse_arch, config, device):
    if sparse_arch:
        if base_arch == "mamba":
            config = EnhancedMambaConfig(**config)
            model = EnhancedMambaModel(config)
        elif base_arch == "mamba2":
            config = EnhancedMamba2Config(**config)
            model = EnhancedMamba2Model(config)
        else:
            raise NotImplementedError
    else:
        if base_arch == "mamba":
            config = MambaConfig(**config)
            model = MambaModel(config)
        elif base_arch == "mamba2":
            config = Mamba2Config(**config)
            model = Mamba2Model(config)
        elif base_arch == "samba":
            config = SambaConfig(**config)
            model = SambaModel(config)
        else:
            raise NotImplementedError

    print("Total Parameters:", sum([para.numel() for para in model.parameters()]))
    return model.to(torch.bfloat16).to(device)

def collate_fn(batch):
    batch_dict = {"inputs": [], "mask": []}
    for data in batch:
        for key in batch_dict:
            batch_dict[key].append(data[key])
    for key in batch_dict:
        batch_dict[key] = torch.tensor(batch_dict[key])
    return batch_dict

class Trainer():
    def __init__(self, model, loss_func, device):
        self.model = model
        self.output_layer = model.get_input_embeddings().weight.T
        self.loss_func = loss_func
        self.device = device

    def forward(self, data, pad_token=3):
        inputs = data["inputs"][:, :-1].to(self.device)
        labels = data["inputs"][:, 1:].to(self.device)
        mask = data["mask"][:, :-1].bool().to(self.device)
        attn_mask = inputs != pad_token
        outputs = self.model(input_ids=inputs, attention_mask=attn_mask, use_cache=False)
        hidden = outputs["last_hidden_state"]
        logits = torch.matmul(hidden, self.output_layer)
        loss = self.loss_func(logits[mask].reshape(-1, logits.shape[-1]), labels[mask].reshape(-1))
        preds = torch.argmax(logits, dim=-1)
        acc = (((preds == labels) & mask).float().sum(-1) / mask.float().sum(-1)).mean()
        gate_loss = outputs["gate_loss"] if hasattr(outputs, "gate_loss") else 0
        return loss, gate_loss, acc

def lr_lambda(current_step):
    if current_step < args.warmup_step:
        return float(current_step) / float(max(1, args.warmup_step))
    return 1.0

def main(args):
    if args.sparse_arch or (args.base_arch == "samba"):
        config = f"{args.config_dir}/{args.base_arch}.json"
    else: # For a fair comparison, we double the hidden size of the base architectures that do not include the sparse attention module.
        config = f"{args.config_dir}/{args.base_arch}_double_size.json"
    config = json.load(open(config))
    config["sparse_arch"] = args.sparse_arch
    config["sparse_keys"] = args.sparse_keys
    config["num_hash"] = args.num_hash
    model_name = f"{args.base_arch}+{args.sparse_arch}_lr_{args.lr}_gate_loss_weight_{args.gate_loss_weight}_seed_{args.seed}"
    wandb.init(config=args, name=model_name)
    dataset = load_dataset("zhan8855/joint-recall-synthetic", cache_dir=args.data_dir)
    dataloader = {split: DataLoader(dataset[split], batch_size=args.bsz, collate_fn=collate_fn) for split in ["train", "validation", "test"]}
    model = load_model(args.base_arch, args.sparse_arch, config, args.device)
    print(model)
    trainer = Trainer(model, torch.nn.CrossEntropyLoss(), args.device)
    optimizer = AdamW(trainer.model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    trainer.model.train()
    best_valid = 0
    best_model = None
    acc_loss = []
    acc_gate_loss = []
    acc_acc = []
    step = 0

    while step < args.step * args.grad_acc:
        for data in tqdm(dataloader["train"]):
            if step == args.step * args.grad_acc:
                break
            step = step + 1

            loss, gate_loss, acc = trainer.forward(data)
            acc_loss.append(loss)
            acc_gate_loss.append(gate_loss)
            acc_acc.append(acc)

            if (step + 1) % args.grad_acc == 0:
                loss = sum(acc_loss) / len(acc_loss)
                gate_loss = sum(acc_gate_loss) / len(acc_loss)
                acc = sum(acc_acc) / len(acc_acc)
                (loss + gate_loss * args.gate_loss_weight).backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                wandb.log({"loss": loss, "gate_loss": gate_loss * args.gate_loss_weight, "train_acc": acc})
                acc_loss = []
                acc_gate_loss = []
                acc_acc = []

            if (step + 1) % (args.grad_acc * args.valid_step) == 0:
                trainer.model.eval()
                total_acc = 0
                with torch.no_grad():
                    for data in tqdm(dataloader["validation"]):
                        loss, gate_loss, acc = trainer.forward(data)
                        total_acc = total_acc + acc.item() * len(data["inputs"])
                print("valid_acc", total_acc / len(dataset["validation"]))
                wandb.log({"valid_acc": total_acc / len(dataset["validation"])})
                if total_acc > best_valid:
                    best_valid = total_acc
                    best_model = deepcopy(trainer.model)
                    torch.save(best_model, f"{args.save_dir}/{model_name}.pt")
                trainer.model.train()

    trainer.model = torch.load(f"{args.save_dir}/{model_name}.pt", weights_only=False)
    trainer.model.eval()
    total_acc = 0
    with torch.no_grad():
        for data in tqdm(dataloader["test"]):
            loss, gate_loss, acc = trainer.forward(data)
            total_acc = total_acc + acc.item() * len(data["inputs"])
    print("test_acc", total_acc / len(dataset["test"]))
    wandb.log({"test_acc": total_acc / len(dataset["test"])})

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
