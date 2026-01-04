import json
import random

def transform(path: str, format="bmes"):
    with open(path, 'r') as f:
        data = f.read()
    data = data.split('\n\n')[:-1]
    print(f"data length: {len(data)}")

    new_data = []
    for d in data:
        if not d:
            continue
        sentence, label = [], []
        for bigram in d.split('\n')[1:]:
            try:
                s, l = bigram.split('\t')
            except:
                print(d)
                print(bigram)
                import pdb;
                pdb.set_trace()
            sentence.append(s)
            label.append(l)

        if format == "bi":
            assert len(sentence) == len(label)
            new_data.append({'sentence': sentence,
                            'label': label})
        elif format == "bmes":
            i = 0
            new_label = []
            while i < len(label):
                if label[i] == 'O':
                    new_label.append('O')
                    i += 1
                elif label[i].startswith("B-"):
                    ent = label[i][2:]
                    start = i
                    i += 1
                    while i < len(label) and label[i] == 'I-' + ent:
                        i += 1
                    end = i
                    if (end - start) == 1:
                        new_label.append('S-' + ent)
                    else:
                        new_label.append('B-' + ent)
                        new_label.extend(['M-' + ent] * (end - start - 2))
                        new_label.append('E-' + ent)
                else:
                    new_label.append("O")
                    i += 1
            try:
                assert len(label) == len(new_label) == len(sentence)
            except:
                import pdb;
                pdb.set_trace()
            new_data.append({'sentence': sentence, 
                            'label': new_label})
        
    with open('unlabeled.json', 'w') as f:
        json.dump(new_data, f, indent=1, ensure_ascii=False)

    # train_size = int(0.8 * len(new_data))
    # val_size = max(1, int(0.1 * len(new_data)))
    # random.shuffle(new_data)
    # train_data = new_data[: train_size]
    # val_data = new_data[train_size: train_size + val_size]
    # test_data = new_data[train_size + val_size:]

    # with open('train.json', 'w') as f:
    #     json.dump(train_data, f, indent=1, ensure_ascii=False)
    # with open('dev.json', 'w') as f:
    #     json.dump(val_data, f, indent=1, ensure_ascii=False)
    # with open('test.json', 'w') as f:
    #     json.dump(test_data, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    transform("unlabeled.txt", "bi")