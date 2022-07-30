# Improving the Trainability of Deep Neural Networks through Layerwise Batch-Entropy Regularization

This is the official source code for the paper: [Improving the Trainability of Deep Neural Networks through Layerwise Batch-Entropy Regularization](https://openreview.net/pdf?id=LJohl5DnZf)

# Setup
To setup the environment simply run the `setup.sh` script. It creates a virtual env. and additionally installs all the requirements needed to run the experiments as provided in the paper.

# Experiments
Every experiment is self-contained i.e. can be used as a code base for future work. In case we executed a hyperparameter search, we also provide the sweep files (`sweep.yaml`) which contain precise hyperparameter setups that were used. Otherwise, a `run.sh` script is provided. For further information how to run a sweep together with an agent we politely refer to the official wandb documentation: https://docs.wandb.ai/guides/sweeps