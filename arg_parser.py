import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=60)
    parser.add_argument('--bert_lr', type=float, default=1e-4)
    parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
    parser.add_argument('--bert_init', type=str, default='bert-base-uncased',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[dataset] if not specified')

    parser.add_argument('--forget-ratio', type=float, default=0.1)
    parser.add_argument('--unlearn-method', type=str, choices=['FT', 'GA', 'FF', 'IU', 'raw', 'retrain'])
    parser.add_argument('--unlearn_lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--unlearn_epochs', default=5, type=int,
                        help='number of total epochs for unlearn to run')
    parser.add_argument('--print_freq', default=20,
                        type=int, help='print frequency')
    return parser.parse_args()