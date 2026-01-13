
import torch
from torch.utils.data import Dataset
import json
import os
from transformers import PreTrainedTokenizer

class LabelSet1D:
    def __init__(self):
        # We explicitly define the set order to ensure stability
        self._labelset = [
            "[PAD]",  # 0
            "O",      # 1
            "B-MISC", "I-MISC",
            "B-PER", "I-PER",
            "B-ORG", "I-ORG",
            "B-LOC", "I-LOC",
            "X",      # 10 (Sub-word alignment token)
            "[CLS]",  # 11
            "[SEP]"   # 12
        ]
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}

    def label2id(self, label: str):
        return self._label2id.get(label, self._label2id["O"]) # Default to O if unknown

    def id2label(self, idx: int):
        return self._id2label.get(idx, '[PAD]')
    
    # --- Dynamic Accessors (Solving the "Magic Number" problem) ---
    @property
    def pad_id(self): return self._label2id["[PAD]"]
    
    @property
    def cls_id(self): return self._label2id["[CLS]"]
    
    @property
    def sep_id(self): return self._label2id["[SEP]"]
    
    @property
    def x_id(self): return self._label2id["X"]

    def __len__(self):
        return len(self._labelset)


import os
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class NERDataset1D(Dataset):
    def __init__(self, dataset_name: str, mode: str, label_set: LabelSet1D, data_type: str = "unlabeled", data_list=None):
        super(NERDataset1D, self).__init__()
        self.data_type = data_type
        self.label_set = label_set

        if data_list is not None:
            self.data = data_list
        else:
            self.path = os.path.join(os.getcwd(), 'data', 'ner', dataset_name, data_type, mode + '.json')
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Dataset not found at {self.path}")
            with open(self.path, 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data[item]
        sentence = row['sentence']
        
        # LOGIC: Check data_type to determine what to return
        if self.data_type == 'labeled':
            # Returns: (sentence, old_ids, new_ids)
            label_old = [self.label_set.label2id(l) for l in row['label_old']]
            label_new = [self.label_set.label2id(l) for l in row['label_new']]
            return sentence, label_old, label_new
        else:
            # Returns: (sentence, label_ids)
            # Default for unlabeled/pre-training data which uses 'label' key
            label = [self.label_set.label2id(l) for l in row['label']]
            return sentence, label


class Collator1D:
    def __init__(self, tokenizer: PreTrainedTokenizer, label_set: LabelSet1D, max_length: int = 128):
        self.tokenizer = tokenizer
        self.label_set = label_set
        self.max_length = max_length

    def _align_labels(self, inputs_encoding, labels, batch_size):
        """
        Helper method to align a list of raw labels to the tokenized subwords.
        Used for both 'labels' (pre-training) and 'targets_old'/'targets_new' (LED).
        """
        aligned_batch = []
        attention_mask = inputs_encoding.attention_mask

        for i in range(batch_size):
            sentence_labels = labels[i]
            word_ids = inputs_encoding.word_ids(batch_index=i)
            
            temp_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special token placeholder
                    temp_labels.append(self.label_set.pad_id)
                elif word_idx != previous_word_idx:
                    # Start of new word
                    temp_labels.append(sentence_labels[word_idx])
                else:
                    # Subword
                    temp_labels.append(self.label_set.x_id)
                
                previous_word_idx = word_idx
            
            # Fix [CLS] and [SEP]
            if len(temp_labels) > 0:
                temp_labels[0] = self.label_set.cls_id
                
                valid_len = attention_mask[i].sum().item()
                if valid_len > 1 and valid_len <= len(temp_labels):
                    temp_labels[valid_len - 1] = self.label_set.sep_id
            
            aligned_batch.append(temp_labels)
            
        return torch.as_tensor(aligned_batch, dtype=torch.long)

    def __call__(self, batch):
        # 1. Unpack based on dataset type
        # Check the length of the first item to see if we have 2 or 3 elements
        elem_len = len(batch[0])
        
        if elem_len == 3:
            # Labeled Dataset: sentence, old, new
            sentences, raw_old, raw_new = map(list, zip(*batch))
        elif elem_len == 2:
            # Unlabeled/Standard Dataset: sentence, label
            sentences, raw_labels = map(list, zip(*batch))
        else:
            raise ValueError(f"Collator expects tuples of length 2 or 3, got {elem_len}")

        # 2. Tokenize (Shared)
        inputs_encoding = self.tokenizer(
            sentences, 
            is_split_into_words=True, 
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = inputs_encoding.input_ids
        attention_mask = inputs_encoding.attention_mask
        batch_size = len(sentences)

        # 3. Align Labels based on type
        if elem_len == 3:
            # LED Mode: Return (input_ids, mask, targets_old, targets_new)
            seq_old = self._align_labels(inputs_encoding, raw_old, batch_size)
            seq_new = self._align_labels(inputs_encoding, raw_new, batch_size)
            return input_ids, attention_mask, seq_old, seq_new
            
        else:
            # Standard Mode: Return (input_ids, mask, labels)
            seq_labels = self._align_labels(inputs_encoding, raw_labels, batch_size)
            return input_ids, attention_mask, seq_labels