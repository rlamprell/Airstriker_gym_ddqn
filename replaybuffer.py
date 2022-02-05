import numpy as np


# Create a Replay Buffer to store  all the states, actions and rewards for all the plays the agent will make
class ReplayBuffer(object):
    def __init__(self, max_size, frame_width, frame_height, frame_channels, n_actions):
        self.memory_size            = max_size
        self.memory_count           = 0

        # Create arrays to store states, next states, actions, rewards and the terminal memorys
        self.state_memory           = np.zeros((self.memory_size, frame_width, frame_height, frame_channels))
        self.next_state_memory      = np.zeros((self.memory_size, frame_width, frame_height, frame_channels))

        # Create blank arrays for storing the actions, rewards and terminal memories
        self.action_memory          = np.zeros((self.memory_size, n_actions), dtype=np.int8)
        self.reward_memory          = np.zeros(self.memory_size)
        self.terminal_memory        = np.zeros(self.memory_size, dtype=np.float32)

    # Store the transition to be used later later in the sample (pulled out in batch size selected)
    def store_transition(self, state, action, reward, state_, done):
        i = self.memory_count % self.memory_size
        self.state_memory[i]        = state
        self.next_state_memory[i]   = state_

        # store one hot encoding of actions
        actions                     = np.zeros(self.action_memory.shape[1])
        actions[action]             = 1.0
        self.action_memory[i]       = actions

        # Store rewards at default values
        self.reward_memory[i]       = reward

        # Flag the terminal states as 0 - these will be ignored when applying the 'learn' function
        self.terminal_memory[i]     = 1 - done

        self.memory_count += 1

    # Pull out random random samples
    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_count, self.memory_size)

        # Choose a random batch from the memory (of size batch_size)
        batch       = np.random.choice(max_mem, batch_size)

        # What are the S, A, R, S_, T in the selected batch
        states      = self.state_memory[batch]
        actions     = self.action_memory[batch]
        rewards     = self.reward_memory[batch]
        states_     = self.next_state_memory[batch]
        terminal    = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal