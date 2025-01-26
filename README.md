<div align="center">

# From Knowledge Forgetting to Accumulation:Evolutionary Relation Path Passing for Lifelong Knowledge Graph Embedding #
</div>

![ERPP](ERPP_overview.pdf)

This is the code release for the paper.

## Quick Start

### Dependencies

```
python==3.8
torch==2.1.0
torchvision==0.11.1
cu118
tqdm
torch-scatter>=2.0.8
pyg==2.4
```

### Train and Test models

0. Switch to `script/` folder
```
cd script/
``` 

1. Run scripts

```
python run.py --gpus 0 --dataset ENTITY --batch_size 32 --dim 64 --layer_num 6 --message_func distmult --aggregate_func sum
```





