import os
import torch

datasets = '20ng,R8,R52,ohsumed,mr'.split(',')
models = 'bert-base-uncased'.split(',')
unlearns = 'raw,retrain,FT,GA,FF,IU'.split(',')
metrics = 'RA,UA,TA'.split(',')

fout = open("result.log", 'w')
for dataset in datasets:
    print(' | '.join(",dataset,model,method".split(',') + metrics + ['']), file = fout)
    print(' | '.join([''] + ['---'] * 6 + ['']), file = fout)
    for model in models:
        for unlearn in unlearns:
            result = [dataset, model, unlearn]
            dir = os.path.join('unlearn_results', dataset, model, unlearn)
            path = os.path.join(dir, 'eval.pkl')
            eval = torch.load(path)
            for metric in metrics:
                if metric == 'UA':
                    result.append("{:.2f}".format(100 - (1 - eval[metric])))
                else:
                    result.append("{:.2f}".format(eval[metric]))
            print(" | ".join([''] + result + ['']), file=fout)
    print(file=fout)