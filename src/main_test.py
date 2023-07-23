# Test the Airstriker bot on the model weights produced on episode 27229 by the Huber MaxPooling Network

from reinforcement_learning.bot import Bot
from reinforcement_learning.ddqn_model import ddqn

# Test the code from here
if __name__ == '__main__':
    # Create bot for the selected game
    airstrikers = Bot(game='Airstriker-Genesis')

    # Get the dimensions of the environment
    frame_height, frame_width, frame_channels = airstrikers.dimensions()

    # Create ddqn model for testing - Greedy to True to evaluate
    deep_model = ddqn(alpha=0.0005, gamma=0.99, n_actions=airstrikers.n_actions, epsilon=0, epsilon_decay=0,
                                 epsilon_minim=0, batch_size=8, frame_width=frame_width, frame_height=frame_height,
                                 frame_channels=frame_channels, greedy=True)

    # Create a greedy version of DDQN and load in the weights from the select .h5 file
    airstrikers.test(runs=101, deep_model=deep_model, test_model_name='src/output/ddqn-models/huber_maxpool-27229-8batches_model.h5')
