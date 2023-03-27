python finetune_bert.py --dataset 20ng --bert_init bert-base-uncased
python unlearn_bert.py --dataset 20ng --unlearn-method raw
python unlearn_bert.py --dataset 20ng --unlearn-method retrain --unlearn_epochs 60
python unlearn_bert.py --unlearn-method FT
python unlearn_bert.py --unlearn-method GA
# python unlearn_bert.py --unlearn-method FF