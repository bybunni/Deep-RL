import numpy as np
import tensorflow as tf
import gym
import time


class PPO:
    """ Single agent implementation of Proximal Policy Optimization with a clipped surrogate objective function,
    Generalized Advantage Estimation, and no shared parameters between the policy (actor) and value function (critic).
    arXiv:1707.06347 arXiv:1506.02438
    """
    def __init__(self, state_size, action_size, max_torque=1, hidden_layers=2, nodes=64, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, epsilon=0.1, c2=0.01, epochs=5, batchsize=64, gamma=0.99, lam=0.95):
        """ Initialization and training parameters for our PPO actor/critic agent. Default values from arXiv:1707.06347
            for continuous control problems.

        Args:
            state_size (int): The number of features in the environment state.
            action_size (int): The number of actions in the environment.
            max_torque (float): The [-max_torque, max_torque] accepted as a valid action by the environment.
            hidden_layers (int): The number of hidden layers for the actor (poilcy) and critic (value) networks.
            nodes (int): The number of nodes in each hidden layer.
            actor_learning_rate (float): Actor Adam optimizer learning rate.
            critic_learning_rate (float): Critic Adam optimizer learning rate.
            epsilon (float): The probability ratio clipping parameter.
            c2 (float): The coefficient for the entropy added to the actor policy loss.
            epochs (int): The number of training epochs per agent update.
            batchsize (int): The size of the minibatches sampled from the rollout.
            gamma (float): The gamma discount value.
            lam (float): The lambda discount used in GAE computation of TD(Lambda) return.
        """

        self.max_torque = max_torque
        self.epochs = epochs
        self.batchsize = batchsize
        self.gamma = gamma
        self.lam = lam

        self.sess = tf.Session()

        def build_actor_critic(action_size, hidden_shape=(hidden_layers, nodes)):
            """

            Args:
                action_size: The number of actions in the environment.
                hidden_shape: (number of hidden layers, number of nodes per hidden layer)

            Returns:
                pi_dist: A Normal distribution actor policy parameterized by learned mu and sigma of dimension action_size.
                pi_parameters: The weights of this actor.
                pi_old_dist: An untrainable copy of our actor to store our old policy.
                pi_old_parameters: The weights of this untrainable actor.
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
            pi_old_dist = build_actor('pi_old', hidden_shape=hidden_shape, trainable=False)

            pi_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
            pi_old_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi_old')

            return pi_dist, pi_parameters, pi_old_dist, pi_old_parameters, v

        self.s = tf.placeholder(tf.float32, shape=[None, state_size], name='state')
        self.v_target = tf.placeholder(tf.float32, shape=[None, 1], name='v_target')

        pi, pi_parameters, pi_old, pi_old_parameters, self.v = build_actor_critic(action_size=action_size)

        self.sample_pi = pi.sample(1)
        self.update_pi_old = [pi_old_parameter.assign(pi_parameter) for pi_parameter, pi_old_parameter
                              in zip(pi_parameters, pi_old_parameters)]

        self.a = tf.placeholder(tf.float32, shape=[None, action_size], name='action')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')

        # log space to prevent underflow ln(a/b) = ln(a) - ln(b)
        probability_ratio = tf.exp(pi.log_prob(self.a) - pi_old.log_prob(self.a))
        # arXiv:1707.06347 Eq. 6
        surrogate = probability_ratio * self.advantage
        # we maximize the objective function by minimizing its negation arXiv:1707.06347 Eq. 7
        actor_loss = -tf.reduce_mean(
            tf.minimum(surrogate, tf.clip_by_value(probability_ratio, 1. - epsilon, 1. + epsilon) * self.advantage))
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
            minibatch_indices = np.random.choice(s.shape[0], size=self.batchsize, replace=False)
            advantages = v_targets[minibatch_indices] - state_values[minibatch_indices]
            self.sess.run(self.actor_optimizer, feed_dict={self.s: s[minibatch_indices],
                                                           self.a: a[minibatch_indices],
                                                           self.advantage: advantages})
            self.sess.run(self.critic_optimizer, feed_dict={self.s: s[minibatch_indices],
                                                            self.v_target: v_targets[minibatch_indices]})
        self.sess.run(self.update_pi_old)

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

    def get_discounted_returns(self, sT, rT, dT, s_, use_gae=True):
        """ Compute either the discounted reward or GAE arXiv:1506.02438 given a rollout of states, rewards, dones and
        the T+1 state.

        Args:
            sT (list): A rollout of states.
            rT (list): A rollout of rewards.
            dT (list): A rollout of dones signifying whether a state is terminal or not.
            s_ (ndarray): The T+1 state.
            use_gae (bool): Whether to return GAE or discounted rewards.

        Returns:
            discounted_returnsT (ndarray): The computed returns for the rollout.
            state_valuesT[:-1]: The current estimate of the state values excluding the T+1 state.
        """
        sT.append(s_)  # we use the T+1 state to compute the target for the Tth state in the rollout
        dT.append(False)  # assume T+1 is non-terminal
        state_valuesT = self.sess.run(self.v, feed_dict={self.s: sT})
        sT.pop()

        if use_gae:
            gaeT = np.zeros_like(rT)
            gae = 0
            for t in reversed(range(len(rT))):
                delta = rT[t] + self.gamma * state_valuesT[t + 1] * (1 - dT[t + 1]) - state_valuesT[t]
                gae = delta + self.gamma * self.lam * (1 - dT[t + 1]) * gae
                gaeT[t] = gae + state_valuesT[t]  # gae + state values is equivalent to a TD(Lambda) advantage estimate
            discounted_returnsT = gaeT
        else:
            discounted_rewardT = np.zeros_like(rT)
            for t in reversed(range(len(rT))):
                if t == len(rT) - 1:
                    discounted_rewardT[t] = rT[t] + self.gamma * state_valuesT[t + 1] * (1 - dT[t + 1])
                else:
                    discounted_rewardT[t] = rT[t] + self.gamma * discounted_rewardT[t + 1] * (1 - dT[t])
            discounted_returnsT = discounted_rewardT

        return discounted_returnsT, state_valuesT[:-1]


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
    agent = PPO(state_size=state_size, action_size=action_size)

    horizon = 128
    episodes = 7500
    t = 0
    sT, aT, rT, dT = [], [], [], []
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
            t += 1
            a = agent.act(s)
            s_, r, done, _ = env.step(a)

            sT.append(s)
            aT.append(a)
            rT.append(r)
            dT.append(done)

            s = s_

            episode_reward += r

            if t % horizon == 0:
                discounted_returnsT, state_valuesT = agent.get_discounted_returns(sT, rT, dT, s_, use_gae=True)
                sT, aT, discounted_returnsT, state_valuesT = \
                    np.vstack(sT), np.vstack(aT), np.vstack(discounted_returnsT), np.vstack(state_valuesT)
                agent.update(sT, aT, discounted_returnsT, state_valuesT)
                sT, aT, rT, dT = [], [], [], []

            if done:
                episode_rewards.append(episode_reward)
                if np.mean(episode_rewards[-100:]) > 200:
                    solved = True
                if episode % 20 == 0:
                    print("\repisode: {}/{}, {} ep average: {:.2f}".format(
                        episode, episodes, len(episode_rewards[-100:]), np.mean(episode_rewards[-100:])),
                        end="", flush=True)

    env.close()
