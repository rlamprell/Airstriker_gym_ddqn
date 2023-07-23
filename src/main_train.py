# Train the Airstriker bot using the Deep Neural Network

from reinforcement_learning.bot import Bot
from reinforcement_learning.ddqn_model import ddqn

# Test the code from here
if __name__ == '__main__':
    # Create bot for the selected game
    airstrikers = Bot(game='Airstriker-Genesis', n_games=50000)

    # Get the dimensions of the environment
    frame_height, frame_width, frame_channels = airstrikers.dimensions()

    # Create ddqn model
    deep_model = ddqn(alpha=0.0005, gamma=0.99, n_actions=airstrikers.n_actions, epsilon=1, epsilon_decay=0.996,
                                 epsilon_minim=0.01, batch_size=8, frame_width=frame_width, frame_height=frame_height,
                                 frame_channels=frame_channels, greedy=False)

    # Train the bot in the selected game using the model
    airstrikers.train(deep_model=deep_model, render=True)
