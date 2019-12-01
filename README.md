# Transformer Based Reinforcement Learning for Games

This repository contains experimental models, written in PyTorch, which incorporate transformers in the Deep Q-Learning tasks, to see if they perform better than the RNN based version (DRQN) or simple DQN.


# Requirements
```
* OpenAI Gym
* PyTorch >= 1.0.0
* Python 3.6+
* Conda (suggested for building environment etc)
* tensorboardx==1.9
* tensorflow==1.14.0 (non-gpu version will do, only needed for tensorboard)

(environment.yml provides detailed list of dependency)
```

# How to run experiments?

Currently, we experiment with the `cartpole` environment, and experiment with 
the three different algorithms, DQN, DRQN (using LSTM) and a transformer based model called DTQN.  

The repo is structured in the following mannner
```
-src/
    |-config_*.py (config files of a particular algorithm)
    |-model_*.py (model definition for a particular algorithm)
    |-train_*.py (training file of a particular algorithm)
    |-memory.py (action replay memory buffer)

-out/
    |-trace_*.txt (traces obtained by different algorithms)
```

To run a particular algorithm (say DQN) one can do ``python train_DQN.py`` this will generate the trace for that algorithm.

# Results

We have performed multiple experiments for each of the algorithms - DQN, DRQN and DTQN. Each algorithm was trained for 5000 episodes and we ran 10 different instances for each of the algorithms with random initialization. <br/> The folowing pictures illustrate the plots of scores over episodes for different runs. 

![Scores vs Episodes for Multiple runs of DQN](https://github.com/udion/Transformer-RL/blob/master/images/DQN_traces.png) <br/>
![Scores vs Episodes for Multiple runs of DRQN](https://github.com/udion/Transformer-RL/blob/master/images/DRQN_traces.png) <br/>
![Scores vs Episodes for Multiple runs of DTQN](https://github.com/udion/Transformer-RL/blob/master/images/DTQN_traces.png) <br/>


