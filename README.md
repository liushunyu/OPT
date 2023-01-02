# Interaction Pattern Disentangling for Multi-Agent Reinforcement Learning

Official codebase for paper [Interaction Pattern Disentangling for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2207.03902). This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) framework and please refer to that repo for more documentation.



## 1. Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.

#### Install StarCraft II

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version 4.10 of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- We use the latest version SC2.4.10 for all SMAC experiments instead of SC2.4.6.2.69232.
- Performance is not comparable across versions.
```

The SMAC maps used for all experiments is in `OPT/src/envs/starcraft2/maps/SMAC_Maps` directory. You should place the `SMAC_Maps` directory in `StarCraftII/Maps`.



## 2. Usage

Please follow the instructions below to replicate the results in the paper.

#### SMAC (Single-Task)
```bash
python src/main.py --config=token_opt --env-config=sc2 with env_args.map_name=[map_name] epsilon_anneal_time=[50000 / 500000] optimizer=['rmsprop' / 'adam']
# map_name:
# 10m_vs_11m (epsilon_anneal_time=50000 optimizer='rmsprop')
# 5m_vs_6m (epsilon_anneal_time=50000 optimizer='rmsprop')
# MMM2 (epsilon_anneal_time=500000 optimizer='rmsprop')
# corridor (epsilon_anneal_time=500000 optimizer='adam')
# 6h_vs_8z (epsilon_anneal_time=500000 optimizer='adam')
# 3s5z_vs_3s6z (epsilon_anneal_time=500000 optimizer='rmsprop')
```

#### SMAC (Multi-Task)

```bash
python src/main.py --config=entity_opt --env-config=sc2mt with env_args.map_name=[map_name] 
# map_name:
# 3-8csz_symmetric
# 3-8MMM_symmetric
# 3-8sz_symmetric
# 5-11csz_symmetric
# 5-11MMM_symmetric
# 5-11sz_symmetric
```

#### Predator-Prey (Multi-Task)

```bash
python src/main.py --config=entity_opt --env-config=prey with env_args.map_name=[map_name] 
# map_name:
# various_cap
# various_num
# various_all
```



## 3. Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2022OPT,
  title={Interaction Pattern Disentangling for Multi-Agent Reinforcement Learning},
  author={Liu, Shunyu and Song, Jie and Zhou, Yihe and Yu, Na and Chen, Kaixuan and Feng, Zunlei and Song, Mingli},
  journal={arXiv preprint arXiv:2207.03902},
  year={2022}
}
```
