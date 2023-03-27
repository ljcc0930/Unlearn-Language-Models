python unlearn_bert.py --dataset 20ng --unlearn-method raw
python unlearn_bert.py --dataset 20ng --unlearn-method retrain --unlearn_epochs 60
python unlearn_bert.py --dataset 20ng --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset 20ng --unlearn-method GA --bert_init bert-base-uncased
python unlearn_bert.py --dataset 20ng --unlearn-method FF --bert_init bert-base-uncased
python unlearn_bert.py --dataset 20ng --unlearn-method IU --bert_init bert-base-uncased --alpha 5

python unlearn_bert.py --dataset R8 --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method GA --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method FF --bert_init bert-base-uncased
python unlearn_bert.py --dataset R8 --unlearn-method IU --bert_init bert-base-uncased --alpha 5

python unlearn_bert.py --dataset R52 --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method GA --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method FF --bert_init bert-base-uncased
python unlearn_bert.py --dataset R52 --unlearn-method IU --bert_init bert-base-uncased --alpha 5

python unlearn_bert.py --dataset ohsumed --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method GA --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method FF --bert_init bert-base-uncased
python unlearn_bert.py --dataset ohsumed --unlearn-method IU --bert_init bert-base-uncased --alpha 5

python unlearn_bert.py --dataset mr --unlearn-method raw --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method retrain --unlearn_epochs 60 --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method FT --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method GA --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method FF --bert_init bert-base-uncased
python unlearn_bert.py --dataset mr --unlearn-method IU --bert_init bert-base-uncased --alpha 5