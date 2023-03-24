import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=60)
    parser.add_argument('--bert_lr', type=float, default=1e-4)
    parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
    parser.add_argument('--bert_init', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[dataset] if not specified')
    return parser.parse_args()