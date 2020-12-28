import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

import sys
sys.path.append("..")

from settings import get_config_by_name
from models.t5_model import T5Model

import pytorch_lightning as pl
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--model-type", type=str, default="unifiedqa")
    parser.add_argument("--checkpoint", type=str, default=None)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("***Using device {}".format(str(device)))

    args = parser.parse_args()
    hparams = get_config_by_name(args.model_type)

    tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_model_name)
    
    if args.checkpoint is not None:
        model = T5Model.load_from_checkpoint(args.checkpoint, args=hparams)
    else:
        model = T5Model(hparams)

    model.to(device)

    with open(args.output_file, 'w') as f:
        print("***Generating outputs for file {}".format(args.input_file))
        print("***Writing to {}".format(args.output_file))
        for line in tqdm(open(args.input_file, 'r').readlines()):
            input_text = line.split('\t')[0]
            inputs = tokenizer(input_text, add_special_tokens=True, max_length=hparams.max_len_in, return_tensors='pt')
            inputs = {key: inputs[key].to(device) for key in inputs}

            output = model.generate(**inputs)[0]
            output_text = tokenizer.decode(output)
            f.write("{}\t{}\n".format(input_text, output_text))
        f.close()

