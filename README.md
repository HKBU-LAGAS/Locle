# Locle

This repository contains code of paper "Leveraging Large Language Models for Effective Label-free Node
Classification in Text-Attributed Graphs". To reproduce the results in the paper, please follow the following steps:

1. Prepare the environment with
```
pip install -r requirements.txt
```
2. The above environment may contains some unrelated packages, you can also just install by yourself with some key requirements:

```
python==3.10.0
torch==2.3.0
torch_cluster==1.6.3
torch_geometric==2.5.3
torch_scatter==2.1.2
torch_sparse==0.6.18
openai==1.59.4
dgl==1.1.1
```

3. Prepare the required datasets. Please download the dataset from [google_drive_link](https://drive.google.com/file/d/1c5yqvUu3PBUhQHvUUlhzs_37IRWcHA2E/view?usp=sharing), and put them under the data/ folder. We have also prepared our annotation cache under the data/annotations/ folder. In most cases, this should be able to satisfy the requirement. If there are still omissions, please change the api url and key and generate the annotations by yourself.

4. Replace the configures in confg.yaml with your own config, and reproduce all the results with 
```
bash train_bash/best_cache.sh
```

# Acknowledgements
This code repo is based on [LLMGNN](https://github.com/CurryTang/LLMGNN). Thanks for their great work!
