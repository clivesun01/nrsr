# Noise-Resilient Symbolic Regression with Dynamic Gating Reinforcement Learning (Code)

This repository is the implementation of the paper [Noise-Resilient Symbolic Regression with Dynamic Gating Reinforcement Learning].

And the Symbolic Regression training framework is based on the study of [Deep Symbolic Optimization].

## Getting started

This code was tested on both `Linux` and `Windows` system:

* Python 3.6
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Step 1: Create a virtual environment with Python 3.6 (here using conda)
```bash
conda create --name nrsr python=3.6
conda activate nrsr
```

Step 2: Install the package for training. Run/Follow steps in [install.sh](install.sh).
```bash
bash install.sh
```

### 2. Training model

Go to the code path (we use the training framework of [Deep Symbolic Optimization]).
```bash
cd ./deep-symbolic-optimization/dso
```

Run an example: Benchmark: Nguyen-1, 10 noisy input variables
```bash
python -m dso.run
```


#### 3. Configuring experiments:

The way to configure the experiments is to modify the parameters in [config_regression.json](codes/config/config_regression.json)

```python
"dataset" : "Nguyen-1",  # the testing benchmark
"n_redundancy": 10,  # the count of noisy input variables added to the original benchmark
```

Replicate experimental can be obtained by running and configuring [run.py](codes/run.py) as follows,

```python
@click.option('--runs', '--r', default=100, type=int, help="Number of independent runs with different seeds")
@click.option('--n_cores_task', '--n', default=1, help="Number of cores to spread out across tasks")
@click.option('--seed', '--s', default=0, type=int, help="Starting seed (overwrites seed in config), incremented for each independent run")
@click.option('--benchmark', '--b', default=None, type=str, help="Name of benchmark")
@click.option('--exp_name', default=None, type=str, help="Name of experiment to manually generate log path")
```
