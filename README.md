# Interaction Pattern Disentangling for Multi-Agent Reinforcement Learning

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2207.03902-b31b1b.svg)](https://arxiv.org/abs/2207.03902)

Official codebase for paper [Interaction Pattern Disentangling for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2207.03902). This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) framework and please refer to that repo for more documentation.

<div align="center">
<img src="https://github.com/liushunyu/OPT/blob/main/introduction.png" width="50%">
</div>

## Overview

**TLDR:** The first work identifies the entangled interaction problem in MARL, and introduce the interaction pattern disentangling task for cooperative MARL, a highly important ingredient for generalization yet largely overlooked by existing literature. Moreover, we propose OPT, a novel disentangling method to decompose entity interactions into interaction prototypes. OPT not only enjoys stronger generalizability, but also exhibits higher interpretability with the disentangled interaction prototypes.

**Abstract:** Deep cooperative multi-agent reinforcement learning has demonstrated its remarkable success over a wide spectrum of complex control tasks. However, recent advances in multi-agent learning mainly focus on value decomposition while leaving entity interactions still intertwined, which easily leads to over-fitting on noisy interactions between entities. In this work, we introduce a novel interactiOn Pattern disenTangling (OPT) method, to disentangle not only the joint value function into agent-wise value functions for decentralized execution, but also the entity interactions into interaction prototypes, each of which represents an underlying interaction pattern within a subgroup of the entities. OPT facilitates filtering the noisy interactions between irrelevant entities and thus significantly improves generalizability as well as interpretability. Specifically, OPT introduces a sparse disagreement mechanism to encourage sparsity and diversity among discovered interaction prototypes. Then the model selectively restructures these prototypes into a compact interaction pattern by an aggregator with learnable weights. To alleviate the training instability issue caused by partial observability, we propose to maximize the mutual information between the aggregation weights and the history behaviors of each agent. Experiments on both single-task and multi-task benchmarks demonstrate that the proposed method yields results superior to the state-of-the-art counterparts.


![image](https://github.com/liushunyu/OPT/blob/main/framework.png)

## Prerequisites

#### Install dependencies

See `requirments.txt` file for more information about how to install the dependencies.

#### Install StarCraft II

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version 4.10 of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- We use the latest version SC2.4.10 for all SMAC experiments instead of SC2.4.6.2.69232.
- Performance is not comparable across versions.
```

The SMAC maps used for all experiments is in `OPT/src/envs/starcraft2/maps/SMAC_Maps` directory. You should place the `SMAC_Maps` directory in `StarCraftII/Maps`.



## Usage

Please follow the instructions below to replicate the results in the paper.

#### SMAC (Single-Task)
```bash
# 10m_vs_11m
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=10m_vs_11m t_max=2050000 epsilon_anneal_time=50000 optimizer='rmsprop'

# 5m_vs_6m
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=2050000 epsilon_anneal_time=50000 optimizer='rmsprop'

# MMM2
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=MMM2 t_max=2050000 epsilon_anneal_time=500000 optimizer='rmsprop'

# corridor
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=corridor t_max=2050000 epsilon_anneal_time=500000 optimizer='adam'

# 6h_vs_8z
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=6h_vs_8z t_max=4050000 epsilon_anneal_time=500000 optimizer='adam'

# 3s5z_vs_3s6z
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z t_max=4050000 epsilon_anneal_time=500000 optimizer='rmsprop'
```

![image](https://github.com/liushunyu/OPT/blob/main/exp-smac-single.png)

#### SMAC (Multi-Task)

```bash
# 3-8csz_symmetric
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=3-8csz_symmetric

# 3-8MMM_symmetric
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=3-8MMM_symmetric

# 3-8sz_symmetric
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=3-8sz_symmetric

# 5-11csz_symmetric
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=5-11csz_symmetric

# 5-11MMM_symmetric
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=5-11MMM_symmetric

# 5-11sz_symmetric
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=5-11sz_symmetric
```

![image](https://github.com/liushunyu/OPT/blob/main/exp-smac-multi.png)

#### Predator-Prey (Multi-Task Training & Zero-Shot Test)

```bash
# various_cap
python src/main.py --config=entity_opt --env-config=prey with env_args.map_name=various_cap

# various_num
python src/main.py --config=entity_opt --env-config=prey with env_args.map_name=various_num

# various_all
python src/main.py --config=entity_opt --env-config=prey with env_args.map_name=various_all
```

![image](https://github.com/liushunyu/OPT/blob/main/exp-pp.png)


## Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2022OPT,
  title={Interaction Pattern Disentangling for Multi-Agent Reinforcement Learning},
  author={Liu, Shunyu and Song, Jie and Zhou, Yihe and Yu, Na and Chen, Kaixuan and Feng, Zunlei and Song, Mingli},
  journal={arXiv preprint arXiv:2207.03902},
  year={2022}
}
```

## Contact

Please feel free to contact me via email (<liushunyu@zju.edu.cn>) if you are interested in my research :)
