python finetune_bert.py --dataset 20ng --bert_init bert-base-uncased

python unlearn_bert.py --dataset 20ng --unlearn-method raw
python unlearn_bert.py --dataset 20ng --unlearn-method retrain --unlearn_epochs 60
python unlearn_bert.py --dataset 20ng --unlearn-method FT
python unlearn_bert.py --dataset 20ng --unlearn-method GA

python finetune_bert.py --dataset R8 --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method GA --bert_init bert-base-uncased

python finetune_bert.py --dataset R52 --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method GA --bert_init bert-base-uncased

python finetune_bert.py --dataset ohsumed --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method GA --bert_init bert-base-uncased

python finetune_bert.py --dataset mr --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method GA --bert_init bert-base-uncased