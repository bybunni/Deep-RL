import numpy as np
import tensorflow as tf
import gym
import time
from collections import deque


class A2C:
    """ Single agent implementation of Advantage Actor Critic (A2C), Generalized Advantage Estimation, and no shared
    parameters between the policy (actor) and value function (critic).
    """
    def __init__(self, state_size, action_size, max_torque=1, hidden_layers=2, nodes=64, actor_learning_rate=2.5e-4,
                 critic_learning_rate=1e-3, c2=0.01, epochs=1, gamma=0.99, lam=0.95, horizon=128):
        """ Initialization and training parameters for our actor/critic agent.

        Args:
            state_size (int): The number of features in the environment state.
            action_size (int): The number of actions in the environment.
            max_torque (float): The [-max_torque, max_torque] accepted as a valid action by the environment.
            hidden_layers (int): The number of hidden layers for the actor (poilcy) and critic (value) networks.
            nodes (int): The number of nodes in each hidden layer.
            actor_learning_rate (float): Actor Adam optimizer learning rate.
            critic_learning_rate (float): Critic Adam optimizer learning rate.
            c2 (float): The coefficient for the entropy added to the actor policy loss.
            epochs (int): The number of training epochs per agent update.
            gamma (float): The gamma discount value.
            lam (float): The lambda discount used in GAE computation of TD(Lambda) return.
        """

        self.max_torque = max_torque
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.horizon = horizon
        self.memory = deque(maxlen=horizon)

        self.sess = tf.Session()

        def build_actor_critic(action_size, hidden_shape=(hidden_layers, nodes)):
            """

            Args:
                action_size: The number of actions in the environment.
                hidden_shape: (number of hidden layers, number of nodes per hidden layer)

            Returns:
                pi_dist: A Normal distribution actor policy parameterized by learned mu and sigma of dimension action_size.
                v: Our value function output.
            """
            def build_hidden_layers(hidden_shape, input):
                number_of_hidden_layers, number_of_nodes = hidden_shape
                last_out = input
                for _ in range(number_of_hidden_layers):
                    last_out = tf.layers.dense(inputs=last_out,
                                               units=number_of_nodes,
                                               activation=tf.nn.relu)
                return last_out

            def build_actor(scope, hidden_shape, trainable):
                with tf.variable_scope(scope):
                    last_hidden_layer = build_hidden_layers(hidden_shape, self.s)
                    mu = self.max_torque * tf.layers.dense(inputs=last_hidden_layer,
                                                           units=action_size,
                                                           activation=tf.nn.tanh,
                                                           trainable=trainable)

                    sigma = tf.layers.dense(inputs=last_hidden_layer,
                                            units=action_size,
                                            activation=tf.nn.softplus,
                                            trainable=trainable)

                    pi_dist = tf.distributions.Normal(loc=mu, scale=sigma)

                return pi_dist

            last_hidden_layer = build_hidden_layers(hidden_shape, self.s)
            v = tf.layers.dense(inputs=last_hidden_layer,
                                units=1,
                                activation=None)  # None is a linear activation

            pi_dist = build_actor('pi', hidden_shape=hidden_shape, trainable=True)

            return pi_dist, v

        self.s = tf.placeholder(tf.float32, shape=[None, state_size], name='state')
        self.v_target = tf.placeholder(tf.float32, shape=[None, 1], name='v_target')

        pi, self.v = build_actor_critic(action_size=action_size)

        self.sample_pi = pi.sample(1)

        self.a = tf.placeholder(tf.float32, shape=[None, action_size], name='action')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')

        actor_loss = -tf.reduce_mean(pi.log_prob(self.a) * self.advantage)
        entropy = c2 * tf.reduce_mean(pi.entropy())
        self.actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate).minimize(actor_loss - entropy)
        critic_loss = tf.reduce_mean(tf.square(self.v_target - self.v))
        self.critic_optimizer = tf.train.AdamOptimizer(critic_learning_rate).minimize(critic_loss)
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, v_targets, state_values):
        """ Given a rollout of states, actions, discounted returns and state values perform an update of the actor
        policy and critic networks given the minibatch size and epochs provided at initialization.

        Args:
            s (ndarray): A rollout of states.
            a (ndarray): A rollout of actions taken.
            v_targets (ndarray): The calculated discounted returns.
            state_values (ndarray): The state values currently estimated by the critic.

        Returns:
            None
        """
        for _ in range(self.epochs):
            advantages = v_targets - state_values
            self.sess.run(self.actor_optimizer, feed_dict={self.s: s,
                                                           self.a: a,
                                                           self.advantage: advantages})
            self.sess.run(self.critic_optimizer, feed_dict={self.s: s,
                                                            self.v_target: v_targets})

    def act(self, s):
        """ Given a single state sample an action from the current policy.

        Args:
            s (ndarray): A single state.

        Returns:
            (ndarray) Float values for each action given the state.
        """
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_pi, feed_dict={self.s: s}).ravel()
        return np.clip(a, -self.max_torque, self.max_torque)

    def get_discounted_returns(self, use_gae=True):
        """ Compute either the discounted reward or GAE arXiv:1506.02438 given the rollout of states, rewards, dones and
        the T+1 state in memory.

        Args:
            use_gae (bool): Whether to return GAE or discounted rewards.

        Returns:
            state_batch (ndarray): The states of the rollout.
            action_batch (ndarray): The actions of the rollout.
            discounted_returnsT (ndarray): The computed returns for the rollout.
            state_valuesT[:-1] (ndarray): The current estimate of the state values excluding the T+1 state.
        """
        memory = np.array(self.memory)
        state_batch = np.vstack(memory[:, 0])
        action_batch = np.vstack(memory[:, 1])
        rT = np.vstack(memory[:, 2]).flatten()
        next_state_batch = np.vstack(memory[:, 3])
        # we use the T+1 state to compute the target for the Tth state in the rollout
        sT = np.vstack((np.vstack(state_batch), next_state_batch[-1]))
        # assume T+1 is non-terminal
        dT = np.vstack((np.vstack(memory[:, 4]).astype(bool), [False])).flatten()

        state_valuesT = self.sess.run(self.v, feed_dict={self.s: sT})

        if use_gae:
            gaeT = np.zeros_like(rT)
            gae = 0
            for t in reversed(range(len(rT))):
                delta = rT[t] + self.gamma * state_valuesT[t + 1] * (1 - dT[t + 1]) - state_valuesT[t]
                gae = delta + self.gamma * self.lam * (1 - dT[t + 1]) * gae
                gaeT[t] = gae + state_valuesT[t]  # gae + state values is equivalent to a TD(Lambda) advantage estimate
            discounted_returnsT = gaeT[:, np.newaxis]
        else:
            discounted_rewardT = np.zeros_like(rT)
            dr = state_valuesT[-1]
            for t in reversed(range(len(rT))):
                discounted_rewardT[t] = dr = rT[t] + self.gamma * dr * (1 - dT[t])
            discounted_returnsT = discounted_rewardT[:, np.newaxis]

        return state_batch, action_batch, discounted_returnsT, state_valuesT[:-1]

    def store_and_learn(self, experience):
        """ Store the provided experience tuple in memory. Once a rollout of length horizon has been collected compute
        the discounted returns, perform an update to the actor and critic, and clear the memory.

        Args:
            experience: A tuple of (state, action, reward, next state, terminal)

        Returns:
            None
        """
        self.memory.append(experience)
        if len(self.memory) == self.horizon:
            states, actions, discounted_returns, state_values = self.get_discounted_returns(use_gae=True)
            self.update(states, actions, discounted_returns, state_values)
            self.memory.clear()


if __name__ == '__main__':
    # Solves LunarLanderContinuous-v2 in < 2000 episodes.
    start = time.time()
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = A2C(state_size=state_size, action_size=action_size)

    episodes = 75000
    episode_rewards = []
    solved = False
    for episode in range(episodes):
        if solved:
            print("\nSolved. 100 Episode Moving Average Reward of {:.2f} in {:.2f} minutes."
                  .format(np.mean(episode_rewards[-100:]), (time.time() - start) / 60))
            break
        episode_reward = 0
        done = False
        s = env.reset()
        while not done:
            a = agent.act(s)
            s_, r, done, _ = env.step(a)

            agent.store_and_learn((s, a, r, s_, done))

            s = s_

            episode_reward += r

            if done:
                episode_rewards.append(episode_reward)
                if np.mean(episode_rewards[-100:]) > 200:
                    solved = True
                if episode % 20 == 0:
                    print("\repisode: {}/{}, {} ep average: {:.2f}".format(
                        episode, episodes, len(episode_rewards[-100:]), np.mean(episode_rewards[-100:])),
                        end="", flush=True)

    env.close()
