import numpy as np
import tensorflow as tf
import gym


class PPO:
    """ A simplified single agent implementation of Proximal Policy Optimization with a clipped surrogate objective
    function. arXiv:1707.06347
    """
    def __init__(self, state_size, action_size, max_torque=1, hidden_nodes=64, actor_learning_rate=1e-4,
                 critic_learning_rate=25e-4, epsilon=0.2, entropy=0.01, epochs=5, batchsize=64, gamma=0.99):

        self.max_torque = max_torque
        self.epochs = epochs
        self.batchsize = batchsize
        self.gamma = gamma

        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, shape=[None, state_size], name='state')

        # Critic
        critic_hidden_layer = tf.layers.dense(inputs=self.s,
                                              units=hidden_nodes,
                                              activation=tf.nn.relu)

        critic_hidden_layer_2 = tf.layers.dense(inputs=critic_hidden_layer,
                                                units=hidden_nodes,
                                                activation=tf.nn.relu)

        self.v = tf.layers.dense(inputs=critic_hidden_layer_2,
                                 units=1,
                                 activation=None)  # None is a linear activation

        self.discounted_reward = tf.placeholder(tf.float32, shape=[None, 1], name='discounted_reward')
        critic_loss = tf.reduce_mean(tf.square(self.discounted_reward - self.v))
        self.critic_optimizer = tf.train.AdamOptimizer(critic_learning_rate).minimize(critic_loss)

        # Actor
        def build_actor(scope, hidden_nodes, action_size, trainable):
            with tf.variable_scope(scope):
                hidden_layer = tf.layers.dense(inputs=self.s,
                                               units=hidden_nodes,
                                               activation=tf.nn.relu,
                                               trainable=trainable)

                mu = self.max_torque * tf.layers.dense(inputs=hidden_layer,
                                                       units=action_size,
                                                       activation=tf.nn.tanh,
                                                       trainable=trainable)

                sigma = tf.layers.dense(inputs=hidden_layer,
                                        units=action_size,
                                        activation=tf.nn.softplus,
                                        trainable=trainable)

                N = tf.distributions.Normal(loc=mu, scale=sigma)

                parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                return N, parameters

        pi, pi_parameters = build_actor(scope='pi',
                                        hidden_nodes=hidden_nodes,
                                        action_size=action_size,
                                        trainable=True)

        pi_old, pi_old_parameters = build_actor(scope='pi_old',
                                                hidden_nodes=hidden_nodes,
                                                action_size=action_size,
                                                trainable=False)

        self.sample_pi = pi.sample(1)
        self.update_pi_old = [pi_old_parameter.assign(pi_parameter) for pi_parameter, pi_old_parameter
                              in zip(pi_parameters, pi_old_parameters)]

        self.a = tf.placeholder(tf.float32, shape=[None, action_size], name='action')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')

        # log space to prevent underflow ln(a/b) = ln(a) - ln(b)
        probability_ratio = tf.exp(pi.log_prob(self.a) - pi_old.log_prob(self.a))
        surrogate = probability_ratio * self.advantage  # Eq. 6
        # we maximize the objective function by minimizing its negation
        actor_loss = -tf.reduce_mean(tf.minimum(
            surrogate,
            tf.clip_by_value(probability_ratio, 1. - epsilon, 1. + epsilon) * self.advantage))  # Eq. 7
        actor_loss += entropy * pi.entropy()  # Eq. 9

        self.actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate).minimize(actor_loss)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, discounted_reward, state_values):
        for K in range(self.epochs):
            minibatch_indices = np.random.choice(s.shape[0], size=self.batchsize, replace=False)
            # Simple advantage estimation https://youtu.be/KHZVXao4qXs?t=4394
            advantages = discounted_reward[minibatch_indices] - state_values[minibatch_indices]
            self.sess.run(self.actor_optimizer, feed_dict={self.s: s[minibatch_indices],
                                                           self.a: a[minibatch_indices],
                                                           self.advantage: advantages})
            self.sess.run(self.critic_optimizer, feed_dict={self.s: s[minibatch_indices],
                                                            self.discounted_reward: discounted_reward[minibatch_indices]})
        self.sess.run(self.update_pi_old)

    def act(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_pi, feed_dict={self.s: s}).ravel()
        return np.clip(a, -self.max_torque, self.max_torque)

    def get_discounted_reward(self, sT, rT, dT):
        state_valuesT = self.sess.run(self.v, feed_dict={self.s: sT})

        discounted_rewardT = np.zeros_like(rT)
        for t in reversed(range(len(rT))):
            if t == len(rT) - 1:
                discounted_rewardT[t] = rT[t] + self.gamma * state_valuesT[t] * (1 - dT[t])
            else:
                discounted_rewardT[t] = rT[t] + self.gamma * discounted_rewardT[t+1] * (1 - dT[t])

        return discounted_rewardT, state_valuesT


if __name__ == '__main__':
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    env.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

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
            print('Solved. 100 Episode Moving Average Reward of {}'.format(np.mean(episode_rewards[-100:])))
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
                discounted_rewardT, state_valuesT = agent.get_discounted_reward(sT, rT, dT)
                sT, aT, discounted_rewardT, state_valuesT = \
                    np.vstack(sT), np.vstack(aT), np.vstack(discounted_rewardT), np.vstack(state_valuesT)
                agent.update(sT, aT, discounted_rewardT, state_valuesT)
                sT, aT, rT, dT = [], [], [], []

            if done:
                episode_rewards.append(episode_reward)
                if np.mean(episode_rewards[-100:]) > 200:
                    solved = True
                if episode % 20 == 0:
                    print("episode: {}/{}, {} ep average: {}".format(
                        episode, episodes, len(episode_rewards[-100:]), np.mean(episode_rewards[-100:])))

    env.close()
