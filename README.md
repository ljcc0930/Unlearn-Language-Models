# Class-wise Unlearning for Language Classification Tasks
Course project for CSE 842.

## Requirements
<!-- [PyTorch](https://pytorch.org/get-started/locally/) [DGL](https://www.dgl.ai/pages/start.html) [PyTorch-Ignite](https://pytorch-ignite.ai/how-to-guides/01-installation/) [Transformer](https://huggingface.co/transformers/v4.2.2/installation.html) -->
```bash
# cuda = 11.8, install with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 ignite nltk scikit-learn dgl transformers -c pytorch -c nvidia -c dglteam/label/cu118 -c huggingface
```
```bash
# cuda = 11.8, install with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install pytorch-ignite transformers scikit-learn nltk
```

## Pre-process data

```bash
bash run_build.sh
```

## Finetune model

```bash
bash run_finetune.sh
```

## Unlearning

```bash
bash run_unlearn.sh
```

## Results
 | dataset | model | method | RA | UA | TA | 
 | --- | --- | --- | --- | --- | --- | 
 | 20ng | bert-base-uncased | raw | 99.83 | 0.20 | 84.03 | 
 | 20ng | bert-base-uncased | retrain | 99.99 | 7.17 | 85.18 | 
 | 20ng | bert-base-uncased | FT | 99.79 | 0.79 | 83.42 | 
 | 20ng | bert-base-uncased | GA | 5.23 | 94.89 | 5.22 | 
 | 20ng | bert-base-uncased | FF | 9.15 | 91.55 | 8.35 | 
 | 20ng | bert-base-uncased | IU | 85.83 | 13.16 | 69.89 | 

 | dataset | model | method | RA | UA | TA | 
 | --- | --- | --- | --- | --- | --- | 
 | R8 | bert-base-uncased | raw | 99.44 | 1.22 | 97.53 | 
 | R8 | bert-base-uncased | retrain | 99.93 | 2.43 | 98.26 | 
 | R8 | bert-base-uncased | FT | 99.77 | 1.01 | 97.62 | 
 | R8 | bert-base-uncased | GA | 0.70 | 98.78 | 0.46 | 
 | R8 | bert-base-uncased | FF | 53.96 | 47.46 | 59.07 | 
 | R8 | bert-base-uncased | IU | 63.95 | 35.50 | 66.51 | 

 | dataset | model | method | RA | UA | TA | 
 | --- | --- | --- | --- | --- | --- | 
 | R52 | bert-base-uncased | raw | 99.87 | 0.00 | 96.61 | 
 | R52 | bert-base-uncased | retrain | 99.89 | 2.90 | 96.53 | 
 | R52 | bert-base-uncased | FT | 99.85 | 0.17 | 95.76 | 
 | R52 | bert-base-uncased | GA | 3.78 | 96.08 | 2.92 | 
 | R52 | bert-base-uncased | FF | 30.52 | 66.44 | 30.26 | 
 | R52 | bert-base-uncased | IU | 99.70 | 0.34 | 96.07 | 

 | dataset | model | method | RA | UA | TA | 
 | --- | --- | --- | --- | --- | --- | 
 | ohsumed | bert-base-uncased | raw | 100.00 | 0.00 | 70.76 | 
 | ohsumed | bert-base-uncased | retrain | 100.00 | 32.78 | 71.88 | 
 | ohsumed | bert-base-uncased | FT | 99.78 | 1.99 | 69.18 | 
 | ohsumed | bert-base-uncased | GA | 17.39 | 80.46 | 14.59 | 
 | ohsumed | bert-base-uncased | FF | 11.21 | 83.77 | 11.82 | 
 | ohsumed | bert-base-uncased | IU | 100.00 | 0.00 | 70.84 | 

 | dataset | model | method | RA | UA | TA | 
 | --- | --- | --- | --- | --- | --- | 
 | mr | bert-base-uncased | raw | 99.65 | 0.16 | 85.79 | 
 | mr | bert-base-uncased | retrain | 100.00 | 13.62 | 85.03 | 
 | mr | bert-base-uncased | FT | 100.00 | 1.41 | 85.68 | 
 | mr | bert-base-uncased | GA | 49.97 | 52.27 | 50.00 | 
 | mr | bert-base-uncased | FF | 59.19 | 40.69 | 57.65 | 
 | mr | bert-base-uncased | IU | 55.39 | 47.42 | 52.34 | 

