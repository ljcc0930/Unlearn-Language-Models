import torch
import time
import os
# import matplotlib.pyplot as plt

# import utils


# def plot_training_curve(training_result, save_dir, prefix):
#     # plot training curve
#     for name, result in training_result.items():
#         plt.plot(result, label=f'{name}_acc')
#     plt.legend()
#     plt.savefig(os.path.join(save_dir, prefix + '_train.png'))
#     plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    print("New checkpoint")
    if args.save_dir is None:
        dir = os.path.join('unlearn_results', args.dataset, args.bert_init, args.unlearn_method)
    else:
        dir = args.save_dir
    os.makedirs(dir, exist_ok=True)
    torch.save(
    {
        'bert_model': model.bert_model.state_dict(),
        'classifier': model.classifier.state_dict(),
        'evaluation_result': evaluation_result,
    },
    os.path.join(dir, "save.pkl")
    )
    torch.save(
    evaluation_result,
    os.path.join(dir, "eval.pkl")
    )


def load_unlearn_checkpoint(model, args):
    if args.save_dir is None:
        dir = os.path.join('unlearn_results', args.dataset, args.bert_init, args.unlearn_method)
    else:
        dir = args.save_dir
    checkpoint_path = os.path.join(dir, "save.pkl")
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint_dict = torch.load(checkpoint_path)
    model.bert_model.load_state_dict(checkpoint_dict['bert_model'])
    model.classifier.load_state_dict(checkpoint_dict['classifier'])
    evaluation_result = checkpoint_dict['evaluation_result']
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.unlearn_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

        # results = OrderedDict((name, []) for name in data_loaders.keys())
        # results['train'] = []

        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()
            print("Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args)
            scheduler.step()

            # results['train'].append(train_acc)
            # for name, loader in data_loaders.items():
            #     print(f"{name} acc:")
            #     val_acc = validate(loader, model, criterion, args)
            #     results[name].append(val_acc)

            # plot_training_curve(results, args.save_dir, args.unlearn)

            print("one epoch duration:{}".format(time.time()-start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
