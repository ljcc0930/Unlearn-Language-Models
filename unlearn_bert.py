import os
import shutil

import torch as th
from torch.optim import lr_scheduler
import torch.utils.data as Data

import unlearn
import arg_parser

from model import BertClassifier
import utils

def main():
    args = arg_parser.parse_args()

    max_length = args.max_length
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    bert_lr = args.bert_lr
    dataset = args.dataset
    bert_init = args.bert_init
    checkpoint_dir = args.checkpoint_dir
    forget_ratio = args.forget_ratio
    unlearn_method = args.unlearn_method

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = utils.load_corpus(dataset)

    nb_node = adj.shape[0]
    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_forget = int(nb_train * forget_ratio)
    nb_retain = nb_train - nb_forget
    nb_class = y_train.shape[1]

    model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class).cuda()

    y = th.LongTensor((y_train + y_val + y_test).argmax(axis=1))

    corpus_file = './data/corpus/'+dataset+'_shuffle.txt'
    with open(corpus_file, 'r') as f:
        text = f.read()
        text=text.replace('\\', '')
        text = text.split('\n')

    def encode_input(text, tokenizer):
        input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        return input.input_ids, input.attention_mask

    input_ids, attention_mask, label = {}, {}, {}

    input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

    # create train/test/val datasets and dataloaders
    curr = 0
    for split, num in zip(['retain', 'forget', 'val', 'test'], [nb_retain, nb_forget, nb_val, nb_test]):
        input_ids[split] = input_ids_[curr: curr + num]
        attention_mask[split] = attention_mask_[curr: curr + num]
        label[split] = y[curr: curr + num]
        curr += num

    datasets = {}
    loader = {}
    for split in ['retain', 'forget', 'val', 'test']:
        datasets[split] =  Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
        loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)
    
    unlearn_func = unlearn.get_unlearn_method(unlearn_method)
    if unlearn_method != "retrain":
        checkpoint = "checkpoint/bert-base-uncased_20ng/checkpoint.pth" # hard coding
        checkpoint_dict = th.load(checkpoint)
        model.bert_model.load_state_dict(checkpoint_dict['bert_model'])
        model.classifier.load_state_dict(checkpoint_dict['classifier'])

    # optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    criterion = th.nn.CrossEntropyLoss()

    unlearn_func(loader, model, criterion, args)
    unlearn.save_unlearn_checkpoint(model, None, args)


if __name__ == "__main__":
    main()
