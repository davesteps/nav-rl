
"""To Do
- other learning algorithms
- learning rate adjust
- experience memory size increase (in the paper they suggest 1m)
- test dueling dqn
- plot training loss

# crossing traffic example
# agent can stop
# increase grid size
# increase size and change shape of target
# geographical realistic region
# calc least cost path to compare with
"""

from routing import *
from utils import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# env = Navigation(grid_size=40)
# env = NavV2(grid_size=30)
env = NavigationV2(grid_size=30,random_land=.05,inc_mvng_hzd=True)
env.max_steps

mem_len = 3

nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=(mem_len,) + env.observation_space.shape))
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
memory = SequentialMemory(limit=500000, window_length=mem_len)

# policy = BoltzmannQPolicy(tau=0.05)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1e7)

agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                 train_interval=4, delta_clip=1.)
agent.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# agent.fit(env, nb_steps=1e4, log_interval=10000,verbose=1)
# agent.save_weights('dqn_nav2_30x30x1_2conv_1e7',overwrite=True)
agent.load_weights('model8_6c01ba5f786ebd5938ca8e8f75008a63d2fe11bd/dqn_nav2_30x30x3_2conv_1e7_mvhz')
#


# Finally, evaluate our algorithm for 20 episodes.
for i in range(1,21):
    print(i)
    agent.test(env, nb_episodes=1,visualize=True)
    makegif('images/','model8_6c01ba5f786ebd5938ca8e8f75008a63d2fe11bd/movie'+str(i),delete_files=True)
