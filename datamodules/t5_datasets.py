from typing import Dict


def seq2seq_preprocess(tokenizer, hparams, x: Dict) -> Dict:
    features = dict()
    features["input"] = tokenizer(x[0], padding='max_length', max_length=hparams.max_len_in, truncation=True, add_special_tokens=True, return_tensors='pt')
    features["decoder_input"] = tokenizer(x[1], padding='max_length', max_length=hparams.max_len_out, truncation=True, add_special_tokens=True, return_tensors='pt')
    return features


class TsvDataset(Dataset):
    def __init__(self, path: str, map_func=None) -> None:
        self.samples = [line.split('\t') for line in open(path, 'r').readlines()]
        self.map_func = map_func

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if self.map_func is None:
            return self.samples[idx]
        else:
            return self.map_func(self.samples[idx])
