# Airstriker_gym_ddqn

## Project Description
This repository trains a Deep Reinforcement Learning agent to play the Sega Genesis game 'Airstriker' using the following tech stack:
- OpenAI's gym retro library.
- Python.
- TensorFlow.

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
