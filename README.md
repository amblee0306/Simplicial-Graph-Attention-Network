# SGAT

SGAT (Simplicial Graph Attention Network) is graph neural model for heterogeneous graph datasets.
This repo supplements our [paper](https://www.ijcai.org/proceedings/2022/0443.pdf).
This version of code is specifically for IMDB dataset.

## Setup
Experiments tested on python3.9 with cuda 10.2 and dgl-cuda 0.6.0.

## How to run

### Parameters
There are two parameters to explicitly change in `train_sgat.py` file. 
- UNINFORMATIVE : Set to True to run with Random Node Features (RNF)
- EDGE_FEATURES : Set to True to run SGAT-EF else it will be SGAT.

### First, create conda virtual environment with the following command 
`conda create --name <env> --file requirements.txt`

### Running the code for IMDB dataset (with GPU)
`python train_sgat.py --dataset IMDB --L 10 --lr 0.005 --num_heads 2 --hidden_units 64 --weight_decay 0.0005`

