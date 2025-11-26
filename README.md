# Code Instruction for GO-Skill

This instruction hosts the PyTorch implementation of ["**Goal-Oriented Skill Abstraction for Offline Multi-Task Reinforcement Learning**"](https://arxiv.org/abs/2507.06628) with the [MetaWorld](https://github.com/Farama-Foundation/Metaworld) benchmark.

**NOTE**: 

The code is based on the [Prompt-DT](https://github.com/mxu34/prompt-dt) codebase.

Since [MetaWorld](https://meta-world.github.io) is under active development,  we perform all the experiments on the stable version: https://github.com/Farama-Foundation/Metaworld/tree/v2.0.0



## Setup

1. Set up the working environment: 

```shell
pip install -r requirements.txt
```

2. Set up the MetaWorld benchmark: 

First, install the mujoco-py package by following the [instructions](https://github.com/openai/mujoco-py#install-mujoco).

Then, install MetaWorld:

```shell
cd envs
pip install -e Metaworld/
```

3. Download the dataset MT50 via this [Google Drive link](https://drive.google.com/drive/folders/1Ce11F4C6ZtmEoVUzpzoZLox4noWcxCEb) from [MTDIFF](https://github.com/tinnerhrhe/MTDiff).



## Training

To train `Prompt-DT` on the `MT50 Near-Optimal` setting, 

```shell
python main_prompt_dt.py --alg_name=prompt_dt --env_name=metaworld with seed=123 env=metaworld-mt50 data_mode=expert
```

To train `Prompt-DT` on the `MT50 Sub-Optimal` setting, 

```bash
python main_prompt_dt.py --alg_name=prompt_dt --env_name=metaworld with seed=123 env=metaworld-mt50 data_mode=medium
```

To train `GO-Skill` on the `MT50 Near-Optimal` setting, 

```shell
python main_skill_dt.py --alg_name=skill_dt --env_name=metaworld with seed=123 env=metaworld-mt50 data_mode=expert
```

To train `GO-Skill` on the `MT50 Sub-Optimal` setting, 

```bash
python main_skill_dt.py --alg_name=skill_dt --env_name=metaworld with seed=123 env=metaworld-mt50 data_mode=medium
```



You can tune any hyperparameters in the `config` folder.



## See Also

See  [Prompt-DT](https://github.com/mxu34/prompt-dt), [Meta-World](https://github.com/Farama-Foundation/Metaworld), [mujoco-py](https://github.com/openai/mujoco-py) for additional instructions.