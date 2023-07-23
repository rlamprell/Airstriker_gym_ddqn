# This is the Double Deep Q-Learning Model
# Here two Neural Networks are created one to select actions and the other to retreive Q-values
# - This helps to avoid the overestimation error found in DQN.

import numpy as np

# Local .py imports
from reinforcement_learning.replaybuffer import ReplayBuffer
from reinforcement_learning.neuralnetwork import build_nn
from tensorflow.keras.models import load_model


# Double Deep Q-Learning Network - The greedy option means always choose the highest Q value
class ddqn(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, epsilon_decay,  epsilon_minim,
                 frame_width, frame_height, frame_channels,
                 memory_size=50000, target_update=100, model_name=None, greedy=False):

        # File name for saving and loading
        self.model_file         = model_name

        # If playing greedily then use the best weights every time
        # In this case the agent should never activate the learn function
        # - however, the minimum is set to 0.0 just in case the user wants to run the learn function
        if greedy:
            self.epsilon        = 0.0
            self.epsilon_minim  = 0.0
        else:
            self.epsilon        = epsilon
            self.epsilon_decay  = epsilon_decay
            self.epsilon_minim  = epsilon_minim

        # How many actions available
        self.action_space       = [i for i in range(n_actions)]
        self.n_actions          = n_actions

        # Batch_size to pick from memory
        self.batch_size         = batch_size

        # Discount Factor
        self.gamma              = gamma

        # Load the memory config - Records all the plays from the environment
        self.memory             = ReplayBuffer(memory_size, frame_width, frame_height, frame_channels, n_actions)

        # Create two DQNs - One to be updated every step, the other every 100
        self.q_evaluation       = build_nn(alpha, n_actions, frame_width, frame_height, frame_channels, "eval_network")
        self.q_target           = build_nn(alpha, n_actions, frame_width, frame_height, frame_channels, "target_network")

        # Update rate for the q_target network
        self.target_update      = target_update

    # Store the values of what just happened
    def store_memory(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # Choose the action to perform - Uses the Evaluation network to make greedy decisions
    def choose_action(self, state):
        state   = state[np.newaxis, :]
        rand    = np.random.random()

        # Epsilon Greedy
        if rand < self.epsilon:
            action  = np.random.choice(self.action_space)
        else:
            actions = self.q_evaluation.predict(state)
            action  = np.argmax(actions)

        return action

    # Make the agent learn
    # Calc Q(s,a) and fit to network
    # Update the target network every set number of steps
    # Decay the epsilon value
    def learn(self):
        # if the memory count if greater than the batch size then update
        if self.memory.memory_count > self.batch_size:
            # Collect a random batch from all stored entries (all arrays of size 'batch size')
            state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

            # Make the new q_target our q_predicted value
            q_target = self.q_evaluation.predict(state)

            # Action selected per batch
            action_indices = np.dot(action, np.array(self.action_space, dtype=np.int8))

            # Batch Index (local 1 - 7)
            batch = np.arange(self.batch_size, dtype=np.int32)

            # Best actions for each batch in the next state
            max_actions = np.argmax(self.q_evaluation.predict(state_), axis=1)

            # RHS Q-Value - Predict the action output for each batch based on the next state
            q_ = self.q_target.predict(state_)

            # q_target Q(s,a) - wraps the reward and actions together (*done will remove everything after end of game)
            q_target[batch, action_indices] = reward + self.gamma*q_[batch, max_actions.astype(int)]*done

            # Fit to eval network using q_target - verbose=0 stops keras printing to console
            self.q_evaluation.fit(state, q_target, verbose=0)

            # Update the target network if we've done 100 steps (update using the eval table)
            # The "100 steps" have been chosen arbitrarily above in the variable 'target_update'
            if self.memory.memory_count % self.target_update == 0:
                self.update_network()

            # Update to the new epsilon value if epsilon has not yet decayed to the lowest allowed value
            if self.epsilon > self.epsilon_minim:
                self.epsilon = self.epsilon * self.epsilon_decay

    # Update the Target Network with the Evaluation Network's weights
    def update_network(self):
        self.q_target.set_weights(self.q_evaluation.get_weights())

    # Save the model
    def save_model(self, i):
        self.q_evaluation.save(f'ddqn-output/{i}-ddqn_model.h5')

    # Load the model
    def load_model(self, model_name):
        self.q_evaluation = load_model(model_name)

        # Load the imported weights into the target network when playing greedily
        if self.epsilon == 0.0:
            self.update_network()