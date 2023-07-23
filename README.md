# Airstriker_gym_ddqn

## Project Description
This repository trains a Deep Reinforcement Learning agent to play the Sega Genesis game 'Airstriker' using the following tech stack:
- OpenAI's gym retro library.
- Python.
- TensorFlow.

## Quick Start
- git clone the repo
- cd into src
- run python main_test.py or python main_train.py

## Architecture



## Results
A collection of videos displaying an Agent's performance at various points.

### MSE 8 batches after 100 episodes
Using MSE, after 100 episodes the agent has begun developing a strategy but is unable to compelte the game.

[View Video](https://github.com/rlamprell/Airstriker_gym_ddqn/assets/90906655/34196079-0858-4cae-8eff-0866cb7b8afe)

### MSE 8 batches after 1278 episodes
Again, using MSE, after 1278 episodes the agent is able to complete the game using a conservative strategy - abusing the edges of game to avoid objects. 

[View Video](https://github.com/rlamprell/Airstriker_gym_ddqn/assets/90906655/b8903e29-ca6c-4d54-b158-5a3ef9463221)

### Huber 8 batches after 27229 episodes
Changing the loss function to Huber and giving the agent significantly more time to play, the agent develops an aggressive playstyle where it attempts to maximise the reward function - shooting as many enemy ships as possible while still completing the game.

[View Video](https://github.com/rlamprell/Airstriker_gym_ddqn/assets/90906655/b4d17162-7a2a-4581-8048-00ce80ba5d6e)


## Languages and Tools:
<p align="left">
  <a href="https://www.python.org" target="_blank" rel="noopener noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>
  </a>
  <a href="https://www.tensorflow.org" target="_blank" rel="noopener noreferrer">
    <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/>
  </a>
</p>


## Next Steps:
- Consolidate the output folder pathing to a variable - currently all over the place.
- Further segment the reinforcement_learning folder - to neural_network, agent, rl_model, etc.
- Introduce linting tool to clean up the general formatting of the code.
- Introduce package manager to make it easier to install - maybe make a pip install.
- Decouple some of the files - similar to the GelSightMujoco repo (probably need to go further than that).
- Optimise the code, could run likely run a lot of this async to increase training speeds.  The batches are probably async already but I could increase the number of batches - I imagine the io between cpu, gpu, ram and ssd are restricting preformance.
- It would be good to be able to turn the renderer on/off during flight.  