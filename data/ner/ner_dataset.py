
import torch
from torch.utils.data import Dataset
import json
import os
from transformers import PreTrainedTokenizer


class LabelSet1D:
    def __init__(self, dataset):
        # postfix = ['train.json', 'dev.json', 'test.json']
        # paths = [os.path.join(os.getcwd(), 'data', 'ner', dataset, p) for p in postfix]
        # self._labelset = set()
        # for p in paths:
        #     with open(p, 'r') as f:
        #         data = json.load(f)
        #     for d in data:
        #         self._labelset.update(set(d['label']))
        # self._labelset = sorted(list(self._labelset), key=lambda x: (x[2:], x[0]))
        self._labelset = ["[PAD]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"] # Fixed label set for Twitter2015
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}

    def label2id(self, label: str):
        return self._label2id[label]

    def id2label(self, idx: int):
        return self._id2label.get(idx, '[PAD]')

    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._labelset)

class NERDataset1D(Dataset):
    def __init__(self, dataset: str, mode: str, label_set: LabelSet1D):
        super(NERDataset1D, self).__init__()
        path = os.path.join(os.getcwd(), 'data', 'ner', dataset, mode + '.json')
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.label_set = label_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence, label = self.data[item]['sentence'], self.data[item]['label']
        label = [self.label_set.label2id(l) for l in label]
        return sentence, label

class Collator1D:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sentences, labels = map(list, zip(*batch))
        inputs_encoding = self.tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True,
                                         max_length=self.max_length)
        input_ids = inputs_encoding.input_ids
        attention_mask = inputs_encoding.attention_mask
        word_ids = [inputs_encoding.word_ids(i) for i in range(len(batch))]

        seq_labels = []
        for ids, las in zip(word_ids, labels):
            temp = [i if i is not None else -100 for i in ids]
            seql = []
            for i in range(len(temp)):
                if temp[i] == -100:
                    # seql.append(-100)
                    seql.append(0)
                else:
                    if temp[i] != temp[i - 1]:
                        seql.append(las[temp[i]])
                    else:
                        # seql.append(-100)
                        seql.append(10)
            seql = [11] + seql[1:-1] + [12]
            seq_labels.append(seql)

        assert len(seq_labels) == len(sentences)
        assert len(seq_labels[0]) == len(input_ids[0])

        return torch.as_tensor(input_ids, dtype=torch.long), \
               torch.as_tensor(attention_mask, dtype=torch.long), \
               torch.as_tensor(seq_labels, dtype=torch.long)