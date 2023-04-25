"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=1000,
            e_greedy_increment=None,
            # e_greedy_increment=0.9/2000,
            output_graph=False,
            double_q=True,
            save_model = 0,
            restore_model = 0
    ):
        self.save_model = save_model
        self.restore_model = restore_model
        self.double_q=double_q
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter = 0

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.q_his = []



    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, n_l3, n_l4, n_l5, n_l6, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 40, 20, 20, 10, 10, 5, \
                tf.variance_scaling_initializer(scale=1.0, mode='fan_in'), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.selu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.selu(tf.matmul(l1, w2) + b2)

            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.selu(tf.matmul(l2, w3) + b3)

            # forth layer. collections is used later when assign to target net
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.selu(tf.matmul(l3, w4) + b4)

            # fifth layer. collections is used later when assign to target net
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, n_l5], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, n_l5], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.selu(tf.matmul(l4, w5) + b5)

            # sixth layer. collections is used later when assign to target net
            with tf.variable_scope('l6'):
                w6 = tf.get_variable('w6', [n_l5, n_l6], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable('b6', [1, n_l6], initializer=b_initializer, collections=c_names)
                l6 = tf.nn.selu(tf.matmul(l5, w6) + b6)

            # seventh layer. collections is used later when assign to target net
            with tf.variable_scope('l7'):
                w7 = tf.get_variable('w7', [n_l6, self.n_actions], initializer=w_initializer, collections=c_names)
                b7 = tf.get_variable('b7', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l6, w7) + b7

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.selu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.selu(tf.matmul(l1, w2) + b2)

            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.selu(tf.matmul(l2, w3) + b3)

            # forth layer. collections is used later when assign to target net
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.selu(tf.matmul(l3, w4) + b4)

            # fifth layer. collections is used later when assign to target net
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, n_l5], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, n_l5], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.selu(tf.matmul(l4, w5) + b5)

            # sixth layer. collections is used later when assign to target net
            with tf.variable_scope('l6'):
                w6 = tf.get_variable('w6', [n_l5, n_l6], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable('b6', [1, n_l6], initializer=b_initializer, collections=c_names)
                l6 = tf.nn.selu(tf.matmul(l5, w6) + b6)

            # seventh layer. collections is used later when assign to target net
            with tf.variable_scope('l7'):
                w7 = tf.get_variable('w7', [n_l6, self.n_actions], initializer=w_initializer, collections=c_names)
                b7 = tf.get_variable('b7', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l6, w7) + b7


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # transition = np.hstack((s, [a, r], s_))

        transition = np.zeros(2*self.n_features+2)
        for i in range(self.n_features):
            transition[i] = s[i]
        transition[self.n_features] = a
        transition[self.n_features+1] = r
        for i in range(self.n_features):
            transition[i+self.n_features+2] = s_[i]

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation, direction):
        # to have batch dimension when feed into tf placeholder
        if self.restore_model == 1:
            self.restore_model = 0
            saver = tf.train.Saver()
            saver.restore(self.sess, 'results_1000day/graph.chkp')
            # self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')


        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # action = np.argmin(actions_value)
            q_temp = -1
            action = -1
            if observation[0][self.n_features-1] >= 0: #up rm
                for i in range(self.n_actions):
                    if (direction[i] == 0) or (direction[i] == 1 and observation[0][i] < observation[0][self.n_features-1]) or (direction[i] == 1 and observation[0][i] == observation[0][self.n_features-1] and observation[0][self.n_actions+i] == 0):
                        if q_temp == -1:
                            q_temp = actions_value[0][i]
                            action = i
                        elif actions_value[0][i] < q_temp:
                            q_temp = actions_value[0][i]
                            action = i
            else:
                for i in range(self.n_actions): # down rm
                    if (direction[i] == 0) or (direction[i] == 1 and observation[0][i] > (observation[0][self.n_features-1]*(-1))) or (direction[i] == 1 and observation[0][i] == (observation[0][self.n_features-1]*(-1)) and observation[0][self.n_actions+i] == 0):
                        if q_temp == -1:
                            q_temp = actions_value[0][i]
                            action = i
                        elif actions_value[0][i] < q_temp:
                            q_temp = actions_value[0][i]
                            action = i
            self.q_his.append(actions_value[0][action])
            # print("actions_value: " + str(actions_value) + " action: " + str(action))
        else:
            action = -1
            if observation[0][self.n_features-1] >= 0: #up rm
                while action == -1 or direction[action] < 0:
                    action = np.random.randint(0, self.n_actions)
            else:
                while action == -1 or direction[action] > 0:
                    action = np.random.randint(0, self.n_actions)
        return action

    ### dqn ###
    # def learn(self):
    #     # check to replace target parameters
    #     if self.learn_step_counter % self.replace_target_iter == 0:
    #         self.sess.run(self.replace_target_op)
    #         # print('\ntarget_params_replaced\n')

    #     # sample batch memory from all memory
    #     if self.memory_counter > self.memory_size:
    #         sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    #     else:
    #         sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    #     batch_memory = self.memory[sample_index, :]

    #     q_next, q_eval = self.sess.run(
    #         [self.q_next, self.q_eval],
    #         feed_dict={
    #             self.s_: batch_memory[:, -self.n_features:],  # fixed params
    #             self.s: batch_memory[:, :self.n_features],  # newest params
    #         })

    #     # change q_target w.r.t q_eval's action
    #     q_target = q_eval.copy()

    #     batch_index = np.arange(self.batch_size, dtype=np.int32)
    #     eval_act_index = batch_memory[:, self.n_features].astype(int)
    #     reward = batch_memory[:, self.n_features + 1]

    #     q_target[batch_index, eval_act_index] = reward + self.gamma * np.min(q_next, axis=1)

    #     """
    #     For example in this batch I have 2 samples and 3 actions:
    #     q_eval =
    #     [[1, 2, 3],
    #      [4, 5, 6]]

    #     q_target = q_eval =
    #     [[1, 2, 3],
    #      [4, 5, 6]]

    #     Then change q_target with the real q_target value w.r.t the q_eval's action.
    #     For example in:
    #         sample 0, I took action 0, and the max q_target value is -1;
    #         sample 1, I took action 2, and the max q_target value is -2:
    #     q_target =
    #     [[-1, 2, 3],
    #      [4, 5, -2]]

    #     So the (q_target - q_eval) becomes:
    #     [[(-1)-(1), 0, 0],
    #      [0, 0, (-2)-(6)]]

    #     We then backpropagate this error w.r.t the corresponding action to network,
    #     leave other action as error=0 cause we didn't choose it.
    #     """

    #     # train eval network
    #     _, self.cost = self.sess.run([self._train_op, self.loss],
    #                                  feed_dict={self.s: batch_memory[:, :self.n_features],
    #                                             self.q_target: q_target})
    #     self.cost_his.append(self.cost)

    #     # increasing epsilon
    #     self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
    #     self.learn_step_counter += 1
    #     print("epsilon: " + str(self.epsilon))

    ### double dqn ###
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmin(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.min(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_this_model(self):
        if self.save_model == 1: 
            saver = tf.train.Saver()
            saver.save(self.sess, 'results/graph.chkp')

    def plot_cost(self):
        import matplotlib.pyplot as plt
        print("len(cost_his): " + str(len(self.cost_his)))
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.annotate(str(self.cost_his[len(self.cost_his)-1]), xy=(len(self.cost_his), self.cost_his[len(self.cost_his)-1]+50), xytext=(len(self.cost_his), self.cost_his[len(self.cost_his)-1]+200),arrowprops=dict(facecolor='black', shrink=0.05))
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def plot_q(self):
        import matplotlib.pyplot as plt
        print("len(q_his): " + str(len(self.q_his)))
        plt.plot(np.arange(len(self.q_his)), self.q_his)
        plt.annotate(str(self.q_his[len(self.q_his)-1]), xy=(len(self.q_his), self.q_his[len(self.q_his)-1]+50), xytext=(len(self.q_his), self.q_his[len(self.q_his)-1]+200),arrowprops=dict(facecolor='black', shrink=0.05))
        plt.ylabel('q')
        plt.xlabel('training steps')
        plt.show()



