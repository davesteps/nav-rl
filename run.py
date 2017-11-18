from routing import Navigation

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from utils import *

env = Navigation(grid_size=10)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=(1,) + env.observation_space.shape))
# model.add(Flatten(input_shape=(3,) + env.observation_space.shape))
# model.add(Conv2D(32, (8, 8), strides=(4, 4)))
# model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy(tau=1.)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# dqn.fit(env, nb_steps=10000, verbose=1)
# dqn.save_weights('dqn_10x10_2m')
dqn.load_weights('model2_1m/dqn_10x10_1m')

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=1,visualize=True)

makegif('images/','movie16')


dqn.test(env, nb_episodes=5)