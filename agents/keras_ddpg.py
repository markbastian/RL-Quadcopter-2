from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    e = np.exp(x)
    ne = np.exp(-e)
    return (e - ne) / (e + ne)
    
def ranged_sigmoid(lo, hi):
    lambda x: sigmoid(x) * (hi - lo) + lo
    
def ranged_tanh(lo, hi):
    lambda x: (tanh(x) + 1.0) * 0.5 * (hi - lo) + lo
    
def capped(lo, hi):
    lambda x: min(max(x, lo), hi)
    
def build_actor(nS, nA, action_min, action_max, print_summary=False):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,nS)))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(16, activation='relu'))
    #actor.add(Dense(nA, activation=ranged_sigmoid(action_min, action_max)))
    actor.add(Dense(nA, activation=capped(action_min, action_max)))
    #actor.add(Dense(nA, activation='linear'))
    if print_summary: 
        print(actor.summary())
    return actor

def build_critic(nS, nA, print_summary=False):
    action_input = Input(shape=(nA,), name='action_input')
    observation_input = Input(shape=(1,nS), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    if print_summary: 
        print(critic.summary())
    return action_input, critic

def build_agent(nS, nA, action_min, action_max):
    actor = build_actor(nS, nA, action_min, action_max)
    action_input, critic = build_critic(nS, nA)
    agent = DDPGAgent(nb_actions=nA, 
                      actor=actor, 
                      critic=critic, 
                      critic_action_input=action_input,
                      memory=SequentialMemory(limit=100000, 
                                              window_length=1), 
                      nb_steps_warmup_critic=100, 
                      nb_steps_warmup_actor=100,
                      random_process=OrnsteinUhlenbeckProcess(size=nA, 
                                                              theta=0.15,
                                                              sigma=0.3), 
                      gamma=0.99, 
                      target_model_update=0.001)
    agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=['mae'])
    return agent