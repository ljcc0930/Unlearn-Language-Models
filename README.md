# CSE-842-proj
Course project for CSE 842.

## Requirements
<!-- [PyTorch](https://pytorch.org/get-started/locally/) [DGL](https://www.dgl.ai/pages/start.html) [PyTorch-Ignite](https://pytorch-ignite.ai/how-to-guides/01-installation/) [Transformer](https://huggingface.co/transformers/v4.2.2/installation.html) -->
```bash
# cuda = 11.8, install with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 ignite nltk scikit-learn cudatoolkit dgl transformers -c pytorch -c nvidia -c dglteam/label/cu118 -c huggingface
```
```bash
# cuda = 11.8, install with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install pytorch-ignite transformers scikit-learn nltk
```
