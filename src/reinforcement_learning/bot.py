# This is the game model
# It creates the agent, who plays the game and tracks all their results
# It is possible to setup both a training and testing environment using the class below

# Package Import
import time as timer
import numpy as np
import os

# Local .py imports
# import reinforcement_learning.retro_wrappers
from reinforcement_learning.retro_wrappers import make_retro, AirstrikerDiscretizer, wrap_deepmind_retro
from utils.plotter import PlotResults

# Make sure there is a folder for outputs, models, videos and a file for logging the outputs
if not os.path.exists('src/output/ddqn-output'):   os.makedirs('src/output/ddqn-output')
if not os.path.exists('src/output/ddqn-models'):   os.makedirs('src/output/ddqn-models')
if not os.path.exists('src/output/ddqn-videos'):   os.makedirs('src/output/ddqn-videos')


# Create a virtual bot and an environment for them to play
# Link the bot to a neural network and track whether or not it's learning as it plays
class Bot:
    def __init__(self, game, n_games=1000, results_filename='src/output/Output.txt', save_video=False):

        # Game name - This can also contain extra arguments, such as which stage to start on
        self.game                = game
        self.n_games             = n_games  # The number of games which will be played

        # Are we logging the q values if so Log the outputs
        self.results_filename    = results_filename

        # Are we saving the video
        self.save_video          = save_video

        # Initialise environment
        self.env, self.n_actions = self.environment_setup()

    # Create the environment using various wrappers to help our bot learn in the DQN
    # Saving the video - Note: this will not replay properly if stochastic frame-skipping is enabled in the wrapper
    def environment_setup(self):
        # Wrap the environment in a modified version of baseline's retro/atari wrappers
        # - If saving the video put the .bk2 files in the ddqn-videos folder
        if self.save_video:
            env = make_retro(game=self.game, record="output/ddqn-videos/", Stochastic_FrameSkip=True)
        else:
            env = make_retro(game=self.game)

        # Limit the number of actions the agent can take (lower is better as there are fewer values to converge on)
        env = AirstrikerDiscretizer(env)

        # Resize the frames received and stack four of them on top of one another - the agent can interpret displacement
        env = wrap_deepmind_retro(env, scale=True, frame_stack=4)

        # Calculate the number of actions the agent can perform
        n_actions = AirstrikerDiscretizer(env).n_actions()

        return env, n_actions

    # Return the shape of the environment as three values so that they can be passed to our network
    def dimensions(self):
        # Reset the environment and get the shape of the state
        input_frame = self.env.reset().shape

        # frame_width, frame_height, frame_channels
        return input_frame[0], input_frame[1], input_frame[2]

    # Reset for next episode
    def reset(self, env):
        # Score and Time are not seen by the agent, but used instead of graphing the performance of the agent
        self.score  = 0
        self.time   = 0

        # initial 'done' and state
        done        = False
        state       = env.reset()

        return state, done

    # Output the results to the console
    def print_results_to_console(self, i, eps_history, ddqn_agent, ddqn_scores, ddqn_times):
        # Update the arrays to be used for graphing results
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(self.score)
        ddqn_times.append(self.time)

        # Average Score and Time output to the console
        avg_score   = np.mean(ddqn_scores   [max(0, i - 20):(i + 1)])
        avg_time    = np.mean(ddqn_times    [max(0, i - 20):(i + 1)])

        # Print the results to the console
        print('episode: ',                    i,
              'score: %.2f'                 % self.score,
              ' average score %.2f'         % avg_score,
              'time: %.2f'                  % self.time,
              ' average time %.2f'          % avg_time,
              )

    # Output to a file
    def print_results_to_file(self, i):
        # Write to the output
        with open(self.results_filename, "a") as myfile:
            myfile.write(f'episode {i}     Score {self.score}     Time {self.time} \n')

    # Train using game model -- Runs the loops for the network and agent tasks
    def train(self, deep_model, render=False, play_slow=False, model_name=None):

        # Load a model if selected
        if model_name is not None:
            deep_model.load_model(model_name)

        # Setup arrays to dump game information
        ddqn_scores = []
        ddqn_times  = []
        eps_history = []

        # Initialise graphing
        plotter = PlotResults()

        # play for i to n+1 games (+1 to ensure the graphs capture the last number)
        for i in range(self.n_games+1):

            # Reset for next run
            state, done = self.reset(self.env)
            previous_lives = 3

            # Run until the agent dies or loses
            # - Note: the "done" event is unlikely to ever trigger on Airstriker
            while not done:

                # Will the agent playing the environment be displayed
                if render:
                    self.env.render()
                    # Slow-down so you can watch it play
                    if play_slow:
                        timer.sleep(0.01)

                # Pick from the number of buttons allowed (Discretizer)
                action = deep_model.choose_action(state)

                # Take the action and record the outputs
                state_, reward, done, info = self.env.step(action)

                # Record the new number of lives
                current_lives = info['lives']

                # If the lives are the same then the agent has not died so process rewards as normal
                if current_lives == previous_lives:
                    # Record the reward so it can be associated to the state action
                    reward = reward

                    # record score and time so that we can analysis the agent performance over time
                    self.score  += reward
                    self.time   += 1

                # Ensure's the agent knows losing/dying is really bad
                elif current_lives < previous_lives:
                    reward = -100

                # Check whether the agent is at the end of an episode or not
                # If the agent lost a life, reached the done condition or the time_steps then break and start another ep
                if current_lives < previous_lives or info['gameover'] == 4:
                    done = 1

                # Save this run's lives for the next state
                previous_lives = current_lives

                # Make the agent remember what's done, seen and the associated rewards with those things
                deep_model.store_memory(state, action, reward, state_, int(done))

                # Make the agent learn from what it now knows
                deep_model.learn()

                # Make the next state the current state
                state = state_

            # Display results in the console
            self.print_results_to_console(i, eps_history, deep_model, ddqn_scores, ddqn_times)
            self.print_results_to_file(i)

            # Save the models of completed games
            if self.time >= 1350:
                # Save the model
                deep_model.save_model(f'{i}--Completed Game')

            # Save the Model and Graph every 100 episodes
            # Also save after the first 10 - just to make sure the model and outputs are working
            if (i == 10) or (i % 100 == 0 and i > 0):
                # Save the model
                deep_model.save_model(i)

                # Save the graph
                x = [i for i in range(i + 1)]
                filename = f'ddqn-output/ddqnRetro-TimeScores--Episodes{i}.png'
                plotter.plot_results(x, ddqn_scores, ddqn_times, filename)

    # Test running game model - loading a model
    def test(self, deep_model, test_model_name, runs=3):

        # Load the model
        deep_model.load_model(test_model_name)

        # Setup arrays to dump game information
        ddqn_scores = []
        ddqn_times  = []
        eps_history = []

        # Loop for the No. of runs selected
        for i in range(runs):

            # Reset for next run
            state, done = self.reset(self.env)
            previous_lives = 3

            # Do until the end of the episode
            # - Note: the "done" event is unlikely to ever trigger on Airstriker
            while not done:
                # Render and make the game play slowly
                self.env.render()
                timer.sleep(0.01)

                # Pick from the number of actions/button combinations allowed
                action = deep_model.choose_action(state)

                # Take the action and record the outputs
                state_, reward, done, info = self.env.step(action)

                # Record the new number of lives
                current_lives = info['lives']

                # If the lives are the same then the agent has not died so process rewards as normal
                if current_lives == previous_lives:
                    # Record the reward so it can be associated to the state action
                    reward = reward

                    # record score and time so that we can analysis the agent performance over time
                    # the -100 reward is not used here as we don't call the learn function while testing
                    self.score  += reward
                    self.time   += 1

                # Check whether the agent is at the end of an episode or not
                # If the agent lost a life, reached the done condition or the time_steps then break and start another ep
                if current_lives < previous_lives or info['gameover'] == 4:
                    done = 1

                # Save this run's lives for the next state
                previous_lives = current_lives

                # Make the next state the current state
                state = state_

            # Display results in the console
            self.print_results_to_console(i, eps_history, deep_model, ddqn_scores, ddqn_times)