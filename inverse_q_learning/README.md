Hi This is an implementation for the paper "Deep Inverse Q-learning with Constraints"

To test the code,

1) Check the conf.yaml file to set all the settings.
2) Run collect_dataset.py to collect demontrations.
>python collect_dataset.py
3) Run train.py to train an agent
>python train.py -algo iql
4) Run eval.py to evaluate an trained agent
>python eval.py


Here I have used the objectworld environment to train the agent. The objectworld enviromnent implementation can be found using this link. 

https://github.com/MatthewJA/Inverse-Reinforcement-Learning
