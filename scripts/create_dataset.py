import os
import sys
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from argparse import ArgumentParser
from tqdm import tqdm

exc_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(exc_dir)

from src.util import set_seed

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to save the generated dataset.")
    parser.add_argument("--max_n_context", type=int, default=16, help="Maximum number of context blocks.")
    parser.add_argument("--max_n_key", type=int, default=16, help="Maximum number of keys per context block.")
    parser.add_argument("--n_value", type=int, default=16, help="Number of distinct values.")
    parser.add_argument("--n_train_per_context_key", default=10000, type=int, help="Number of training datapoints to generate for each (n_context, n_key) pair. \
                                                                                    The script enumerates n_context in the range [5, 'max_n_context'] and n_key in the range [5, 'max_n_key']. \
                                                                                    For each (n_context, n_key) pair, 'n_train_per_context_key' datapoints are generated.")
    parser.add_argument("--n_test_per_context_key", default=100, type=int, help="Number of validation and test datapoints to generate for each (n_context, n_key) pair.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()
    return args

def generate_data_item(args, gen1_len, gen2_len, bos_token=1, sep_token=4, pad_token=3, bias=10):
    pad_len = (args.max_n_context * 2) * (args.max_n_key * 2 + 1) + 2
    inputs, mask = [], []
    matrix = np.random.randint(0, args.n_value, (gen1_len, gen2_len))
    inputs.append(bos_token)
    mask.append(0)
    for idx1 in np.random.permutation(gen1_len):
        inputs.append(args.max_n_key + args.n_value + idx1 + bias)
        mask.append(0)
        for idx2 in np.random.permutation(gen2_len):
            inputs.append(args.n_value + idx2 + bias)
            mask.append(0)
            inputs.append(matrix[idx1][idx2] + bias)
            mask.append(0)
    inputs.append(sep_token)
    mask.append(0)
    for idx1 in np.random.permutation(gen1_len):
        inputs.append(args.max_n_key + args.n_value + idx1 + bias)
        mask.append(0)
        for idx2 in np.random.permutation(gen2_len):
            inputs.append(args.n_value + idx2 + bias)
            mask.append(1)
            inputs.append(matrix[idx1][idx2] + bias)
            mask.append(0)
    cur_len = len(inputs)
    assert cur_len <= pad_len
    inputs.extend([pad_token] * (pad_len - cur_len))
    mask.extend([0] * (pad_len - cur_len))
    inputs_str = " ".join(map(str, inputs))
    mask_str = " ".join(map(str, mask))
    return np.array(inputs), np.array(mask), inputs_str, mask_str

def prepare_data(args):
    all_data = set()
    for split, size in zip(["train", "valid", "test"], [args.n_train_per_context_key, args.n_test_per_context_key, args.n_test_per_context_key]):
        inputs_file = open(f"{args.data_dir}/max_n_context_{args.max_n_context}_max_n_key_{args.max_n_key}_seed_{args.seed}_{split}_inputs.txt", "w")
        mask_file = open(f"{args.data_dir}/max_n_context_{args.max_n_context}_max_n_key_{args.max_n_key}_seed_{args.seed}_{split}_mask.txt", "w")
        for idx in tqdm(range(size)):
            for gen1_len in range(5, args.max_n_context + 1):
                for gen2_len in range(5, args.max_n_key + 1):
                    while True:
                        inputs, mask, inputs_str, mask_str = generate_data_item(args, gen1_len, gen2_len)
                        if inputs_str not in all_data:
                            all_data.add(inputs_str)
                            break
                    print(inputs_str, file=inputs_file)
                    print(mask_str, file=mask_file)
        inputs_file.close()
        mask_file.close()

def iter_lines(inputs_path, mask_path):
    with open(inputs_path, "r") as fin_in, open(mask_path, "r") as fin_mask:
        for lin_in, lin_mask in zip(fin_in, fin_mask):
            inputs = list(map(int, lin_in.strip().split()))
            mask = list(map(int, lin_mask.strip().split()))
            yield {"inputs": inputs, "mask": mask}

def convert_to_hf_dataset(args):
    splits = {}
    for split, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        inputs_path = f"{args.data_dir}/max_n_context_{args.max_n_context}_max_n_key_{args.max_n_key}_seed_{args.seed}_{split}_inputs.txt"
        mask_path = f"{args.data_dir}/max_n_context_{args.max_n_context}_max_n_key_{args.max_n_key}_seed_{args.seed}_{split}_mask.txt"
        features = Features({"inputs": Sequence(Value("int32")), "mask": Sequence(Value("int8"))})
        gen = lambda: iter_lines(inputs_path, mask_path)
        splits[hf_split] = Dataset.from_generator(gen, features=features)
    dataset = DatasetDict(splits)
    dataset.save_to_disk(args.data_dir)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    prepare_data(args)
    convert_to_hf_dataset()