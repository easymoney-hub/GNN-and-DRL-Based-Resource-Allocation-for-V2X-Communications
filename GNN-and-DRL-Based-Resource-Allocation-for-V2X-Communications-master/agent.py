from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import networkx as nx
import os
from Environment import *
import matplotlib.pyplot as plt
from model_Graph import GraphModel
from Graph_SAGE import GraphSAGE_sup
from base import BaseModel
from dqn_model import DQNModel
from replay_memory import ReplayMemory
from tensorflow.python import debug as tf_debug

#测试
class Agent(BaseModel):
    def __init__(self, config, environment):
        self.weight_dir = 'weight'
        self.env = environment
        self.G = GraphSAGE_sup(environment)
        self.dqn = DQNModel()
        self.dqn.compile_model()
        #self.history = History(self.config)
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir) 
        self.max_step = 100000
        self.RB_number = 20
        self.num_vehicle = 40
        print('-------------------------------------------')
        print(self.num_vehicle)
        print('-------------------------------------------')
        #[self.num_vehicle, 3, 2]---3表示每个车辆有3个邻居，2表示信道选择（0到self.RB_number）和功率选择（23 dB, 10 dB, 5 dB）
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'int32')   # 训练时 this is actions that taken by V2V links with power
        self.reward = []
        self.learning_rate = 0.01 #初始学习率为 0.01
        self.learning_rate_minimum = 0.0005 #学习率的最小值
        self.learning_rate_decay = 0.96 #学习率的衰减率
        self.learning_rate_decay_step = 500000 #学习率衰减的步数
        self.target_q_update_step = 100 #目标Q网络更新的步数
        self.discount = 0.5 #折扣因子
        self.double_q = True #使用双 Q 网络
        print("------------")
        print(self.double_q)
        print("------------")
        #self.build_dqn()
        self.V2V_number = 3 * len(self.env.vehicles)    # every vehicle need to communicate with 3 neighbors
        #print('self.V2V_number', self.V2V_number)
        self.training = True
        self.GraphSAGE = True
        self.channel_reward_save = np.zeros((60, 20)) #存储信道奖励的历史信息
        self.channel_reward = np.zeros((60, 20)) #存储当前信道奖励信息
        self.neighbor_nodes = []  #存储邻居节点信息
        #self.actions_all = np.zeros([len(self.env.vehicles),3], dtype = 'int32')

    # 将action分割（信道和功率），并将这些值存储在action_all_with_power数组中--信道选择动作记录在[i,j,0]，功率选择动作记录在[i,j,1]--i，j分别是车辆和邻居的索引
    def merge_action(self, idx, action):
        self.action_all_with_power[idx[0], idx[1], 0] = action % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action/self.RB_number))

    # 从环境中获取特定索引idx（车辆及其邻居，输入是一个【i，j】列表）对应的状态信息
    def get_state(self, idx):
    # ===============
    #  Get State from the environment
    # =============
        vehicle_number = len(self.env.vehicles)
        #根据索引获取V2V通道信息、V2I通道信息、V2V干扰信息，并对其进行归一化处理？（缩放）
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],self.env.vehicles[idx[0]].destinations[idx[1]],:] - 80)/60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80)/60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0], idx[1], :] - 60)/60
        # 记录邻居节点的选择情况，初始化为零
        NeiSelection = np.zeros(self.RB_number)
        "选择邻居这里没看懂"
        neighs_list = self.neighbor_nodes[3*idx[0]+idx[1]]  #得到的是嵌套列表--如[[1, 4, 8]]   每个数字代表一个通信链路的编号？？
        neighs_list = neighs_list[0] # 第二次赋值通过[0]去除一层嵌套--得到 [1, 4, 8]
        # print('neighs_list', neighs_list)
        # print('link', self.G.link.dtype)
        for neigh_idx in neighs_list:
            if self.training:
                # print('idx', neigh_idx)
                # print('索引', self.G.link[neigh_idx, 1]) 表示通信发起的车辆
                # print('第几个目的地', self.G.link[neigh_idx, 1]%3) 通信链路中的目标车辆，表示接收通信的车辆
                # 当会话建立时(发送车到连接车)，其对应位置元素更新为1，表示该信道被选择
                NeiSelection[self.action_all_with_power_training[self.G.link[neigh_idx, 0], self.G.link[neigh_idx, 1] % 3, 0]] = 1
            else:
                NeiSelection[self.action_all_with_power[self.G.link[neigh_idx, 0], self.G.link[neigh_idx, 1] % 3, 0]] = 1
        # NeiSelection = NeiSelection / np.max(NeiSelection)
        # 剩余时间--当前车辆需求/总需求？  （剩余需要发送的比特与总需要发送的比特之间的比率）
        """这样取名是否合理？"""
        time_remaining = np.asarray([self.env.demand[idx[0],idx[1]] / self.env.demand_amount])
        #剩余时间负载 （延迟约束下的剩余传输时间）
        load_remaining = np.asarray([self.env.individual_time_limit[idx[0],idx[1]] / self.env.V2V_limit])
        #print('shapes', time_remaining.shape,load_remaining.shape)
        #V2I_channel 20维, V2V_interference 20维, V2V_channel 20维, NeiSelection 20维, time_remaining 1维, load_remainin 1维---总维度 82
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, NeiSelection, time_remaining, load_remaining)) #,time_remaining))
        #return np.concatenate((V2I_channel, V2V_interference, V2V_channel, time_remaining, load_remaining))#,time_remaining))

    #根据输入的状态s_t选择动作，策略是epsilon-greedy
    def predict(self, s_t,  step, test_ep = False):
        # ==========================
        #  Select actions
        # ======================
        ep = 1/(step/20000 + 1)
        if random.random() < ep and test_ep == False:   # epsion to balance the exporation and exploition
            action = np.random.randint(60)  #动作空间为60
        else:
            #print('predict')
            q = self.dqn.forward(s_t)
            action_tensor = tf.argmax(q, axis=1)
            action = action_tensor.numpy()
            #action =self.q_action.eval({self.s_t:[s_t]})[0]
        return action

    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training 
        # ---------
        self.memory.add(prestate, state, reward, action) #将经验添加到记忆缓冲区
        #print(self.step)
        if self.step > 0:
            if self.step % 50 == 0: #每50步进行一次小批量训练
                #print('Training')
                self.q_learning_mini_batch()            # training a mini batch
                #self.save_weight_to_pkl()
                self.dqn.model.save_weights('weight/dqn_weights.h5')
                #self.G.save_graph_network_weights()
            # if self.step % self.target_q_update_step == self.target_q_update_step - 1:
            #     #print("Update Target Q network:")
            #     self.dqn.update_target_network()           # ?? what is the meaning ??
            #     print('update_target_network')

    # 将局部观察与通过图神经网络（GraphSAGE）学习的全局信息相结合，增强了每辆车的状态表示
    # 初始化更好的状态表示，使用图神经网络（GraphSAGE）进行嵌入，返回改善后的状态表示
    def initial_better_state(self, step, Graph_SAGE_label = True):
        self.G.num_V2V_list = np.zeros((len(self.env.vehicles), len(self.env.vehicles)))
        #print("self.num_vehicle", len(self.env.vehicles))
        #print("num_V2V_list.shape", num_V2V_list.shape)
        # 存储每个车辆与其邻居通信时节点特征（node_f）和状态信息(state_old)的数组
        node_f = np.zeros((3*len(self.env.vehicles), 60))
        state_old = np.zeros((3 * len(self.env.vehicles), 82))
        idx = []
        neighs_idx = []
        label = np.zeros(20)
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                self.G.num_V2V_list[i, self.env.vehicles[i].destinations[j]] = 1
        self.G.link = np.zeros((3 * len(self.env.vehicles), 2))  #初始化链表为零，表示最初没有通信
        self.G.features = np.zeros((3 * len(self.env.vehicles), 60))     #初始化车辆之间每个链接的特征矩阵
        [graph, order_nodes, _] = self.G.build_graph(self.G.num_V2V_list)
        # print('order_nodes', order_nodes)
        # print('len(order_nodes)', len(order_nodes))
        self.G.features = np.zeros((3*len(self.env.vehicles), 60))
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                neighs_idx = []
                node_label = order_nodes[3 * i + j]
                neighs = list(nx.neighbors(graph, node_label))
                neighs_idx.append([order_nodes.index(neigh) for neigh in neighs])
                self.neighbor_nodes.append(neighs_idx)
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                state_old[3*i+j, :] = self.get_state([i, j])
                self.G.features[3*i+j, :] = state_old[3*i+j, :60]
                # self.G.num_V2V_list[i, self.env.vehicles[i].destinations[j]] = 1
                idx.append(3*i+j)
        self.channel_reward = state_old[:, 0:20] + state_old[:, 20:40] - state_old[:, 40:60]
        max_value = np.max(self.channel_reward)
        # 计算比例因子
        scale_factor = max_value / 0.8
        # 将数组中的每个值除以比例因子
        self.channel_reward = self.channel_reward / scale_factor
        # [graph, order_nodes, _] = self.G.build_graph(self.G.num_V2V_list)
        self.G.load_graph(graph, order_nodes)
        node_embeddings =self.G.use_GraphSAGE(self.channel_reward, step, idx, Graph_SAGE_label)
        max_value = np.max(node_embeddings)
        # 计算比例因子
        scale_factor = max_value
        # 将数组中的每个值除以比例因子
        node_embeddings = node_embeddings / (scale_factor+0.0001)
        #print('node_embeddings',node_embeddings)
        better_state = np.concatenate((node_embeddings, state_old), axis=1)
        # if Graph_SAGE_label == False :
        #     mask = np.array([1] * state_old.shape[1] + [0] * node_embeddings.shape[1])
        #     better_state = better_state * mask
        return better_state

    def train(self):
        self.dqn.update_target_network()
        #self.G = GraphSAGE_sup(self.env)
        num_game, self.update_count, ep_reward = 0, 0, 0. #游戏轮数，更新次数，回合（episode）内的奖励总和
        total_reward, self.total_loss, self.total_q = 0.,0.,0. #总奖励，总损失，总Q值
        idx_renew_environment = 0 #记录重置环境的次数
        renew_bound_rate = 150 #性能阈值，评估V2I的通信速率
        renew_bound_p = 0.05 #概率阈值，用于评估通信失败率
        max_avg_ep_reward = 0
        ep_reward, actions = [], []        
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        self.env.new_random_game(20)#车辆数目二十
        self.GraphSAGE = True
        better_state = self.initial_better_state(0, self.GraphSAGE)
        for self.step in (range(0, 50001)): # need more configuration
            #训练开始时初始化
            if self.step == 0:                   # initialize set some varibles
                num_game, self.update_count,ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []
            # prediction
            # action = self.predict(self.history.get())
            if (self.step % 2000 == 1 and self.step > 1): #每两千轮评估一次环境中agent的性能
                idx_renew_environment = 0
                # test_rate, percent = self.test_environment()  # 测试环境，获取 test_rate
                # n1 = np.minimum(renew_bound_rate + 10, 165)
                # n2 = np.maximum(renew_bound_p - 0.01, 0.025)
                # print('n1', n1)
                # print('n2', n2)
                # if (test_rate > n1 and percent < n2):
                while True:
                    print('renew environment')
                    self.env.new_random_game(20)  # 假设这是重置环境的操作
                    idx_renew_environment += 1
                    better_state = self.initial_better_state(0, self.GraphSAGE)  # 初始化更好的状态
                    test_rate, percent = self.test_environment()  # 测试环境，获取 test_rate
                    # 如果 test_rate 大于 160，则继续循环
                    if test_rate <= renew_bound_rate or percent > renew_bound_p:
                        break  # 如果满足条件，退出循环
                    if idx_renew_environment > 5: #如果五次环境测试后还未达到要求，放松对环境性能的要求
                        idx_renew_environment = 0
                        renew_bound_rate += 5 
                        renew_bound_p -= 0.01
                        print('renew_bound_rate', renew_bound_rate)
                        print('renew_bound_p', renew_bound_p)

            # if (self.step % 200 ==0):#设置每多少步切换一次智能体获取的状态（有或没有来自GraphSAGE的信息）
                # self.GraphSAGE = not self.GraphSAGE
                # print('self.GraphSAGE', self.GraphSAGE)
            print(self.step)
            # state_old = self.get_state([0, 0])
            # print("state", state_old)
            self.training = True
            self.GraphSAGE = True

            for k in range(1):  # 现在需要先一个循环获取车辆的旧的状态，然后将输出结构送到BS，然后得到状态，进行动作
                rewards = []
                actions_train = []
                Graph_losses = []
                loss_G = []
                # state_old = np.zeros((60, 102))
                for i in range(len(self.env.vehicles)):              
                    for j in range(3):#遍历三个邻居
                        idx = []
                        label = np.zeros(20)
                        idx.append(3*i+j) #当前车辆-邻居对的唯一索引
                        # state_old = self.get_state([i,j])
                        # print(better_state)
                        state_old = self.get_state([i, j])  # 获取当前状态
                        self.G.num_V2V_list[i, self.env.vehicles[i].destinations[j]] = 1  # 标记连接
                        self.G.features[3*i+j, :] = state_old[:60] #更新节点特征
                        
                        #使用GraphSAGE生成节点嵌入
                        node_embeddings = self.G.use_GraphSAGE(self.channel_reward, self.step, idx, self.GraphSAGE)
                        max_value = np.max(node_embeddings)
                        # 计算比例因子
                        scale_factor = max_value
                        # 归一化嵌入向量 将数组中的每个值除以比例因子 
                        node_embeddings = node_embeddings / (scale_factor + 0.0001)
                        node_embeddings = np.squeeze(node_embeddings)
                        # print('node_embeddings',node_embeddings.shape)
                        # print('state_old[3*i+j]', state_old.shape)
                       
                       #组合GraphSAGE嵌入和原始状态
                        better_state_old = np.concatenate((node_embeddings, state_old), axis=0)
                        action = self.predict(better_state_old, self.step) # 基于增强状态预测动作
                        # self.merge_action([i,j], action)
                        # 将动作分解为信道选择和功率选择
                        self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action/self.RB_number))
                        #执行动作，更新并记录奖励
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i,j])
                        self.channel_reward[3*i+j, action % self.RB_number] = reward_train
                        actions_train.append(action)
                        rewards.append(reward_train)
                        state_new = self.get_state([i, j]) # 获取执行动作后的新状态
                        self.G.features[3 * i + j, :] = state_new[:60]
                        node_embeddings_new = self.G.use_GraphSAGE(self.channel_reward, self.step, idx, self.GraphSAGE) # 为新状态生成GraphSAGE嵌入
                        max_value = np.max(node_embeddings_new)
                        # 计算比例因子
                        scale_factor = max_value
                        # 将数组中的每个值除以比例因子
                        node_embeddings_new = node_embeddings_new / (scale_factor + 0.0001)
                        node_embeddings_new = np.squeeze(node_embeddings_new)
                        # print('node_embeddings', node_embeddings_new.shape)
                        # print('state_old[3*i+j]', state_old.shape)
                        better_state_new = np.concatenate((node_embeddings_new, state_new), axis=0) # 组合新状态
                        # 存储经验(state, action, reward, next_state)
                        self.observe(better_state_old, better_state_new, reward_train, action)
                        # print('reward', reward_train)
                if self.step % self.target_q_update_step == self.target_q_update_step - 1 and self.step > 0:
                    # print("Update Target Q network:")
                    self.dqn.update_target_network()  # ?? what is the meaning ??
                    print('update_target_network')
                used_blocks = np.unique(self.action_all_with_power_training[:, :, 0])
                num_used_blocks = len(used_blocks)
                print('num_used_blocks', num_used_blocks)


            # if (self.step % 2000 == 0) and (self.step > 0):
            #     # testing
            #     self.training = False
            #     self.GraphSAGE = False
            #     number_of_game = 10
            #     if (self.step % 10000 == 0) and (self.step > 0):
            #         number_of_game = 50
            #     if (self.step == 38000):
            #         number_of_game = 100
            #     V2I_Rate_list = np.zeros(number_of_game)
            #     Fail_percent_list = np.zeros(number_of_game)
            #     for game_idx in range(number_of_game):
            #         print('self.num_vehicle_test', self.num_vehicle)
            #         self.env.new_random_game(self.num_vehicle)
            #         better_state = self.initial_better_state(0, self.GraphSAGE)
            #         #self.GraphSAGE = not self.GraphSAGE
            #         #print('GraphSAGE_label', self.GraphSAGE)
            #         test_sample = 200
            #         Rate_list = []
            #         print('test game idx:', game_idx)
            #         for k in range(test_sample):
            #             action_temp = self.action_all_with_power.copy()
            #             for i in range(len(self.env.vehicles)):
            #                 self.action_all_with_power[i, :, 0] = -1
            #                 sorted_idx = np.argsort(self.env.individual_time_limit[i,:])
            #                 for j in sorted_idx:
            #                     #state_old = self.get_state([i,j])
            #                     idx = []
            #                     idx.append(3 * i + j)
            #                     state_old = self.get_state([i, j])
            #                     self.G.features[3 * i + j, :] = state_old[:60]
            #                     node_embeddings = self.G.use_GraphSAGE(self.channel_reward, self.step, idx,
            #                                                            self.GraphSAGE)
            #                     max_value = np.max(node_embeddings)
            #                     # 计算比例因子
            #                     scale_factor = max_value
            #                     # 将数组中的每个值除以比例因子
            #                     node_embeddings = node_embeddings / (scale_factor + 0.0001)
            #                     node_embeddings = np.squeeze(node_embeddings)
            #                     # print('node_embeddings',node_embeddings.shape)
            #                     # print('state_old[3*i+j]', state_old.shape)
            #                     better_state_old = np.concatenate((node_embeddings, state_old), axis=0)
            #                     action = self.predict(better_state_old, self.step, True)
            #                     self.merge_action([i, j], action)
            #                 if i % (len(self.env.vehicles)/5) == 1:
            #                     action_temp = self.action_all_with_power.copy()
            #                     reward, percent = self.env.act_asyn(action_temp) #self.action_all)
            #                     Rate_list.append(np.sum(reward))
            #             #print("actions", self.action_all_with_power)
            #         V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            #         Fail_percent_list[game_idx] = percent
            #         #print("action is", self.action_all_with_power)
            #         print('failure probability is, ', percent)
            #         print('mean_V2I_Rate', V2I_Rate_list[game_idx])
            #         #print('action is that', action_temp[0,:])
            # #print("OUT")
            #     self.dqn.model.save_weights('weight/dqn_weights.h5')
            #     #self.G.save_graph_network_weights()
            #     print('The number of vehicle is ', len(self.env.vehicles))
            #     print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
            #     print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
            #     #print('Test Reward is ', np.mean(test_result))
            # if self.step == 5000:
            #     # 绘制损失曲线
            #     plt.plot(self.G.loss)
            #     plt.title('Loss over time')
            #     plt.xlabel('Epoch')
            #     plt.ylabel('Loss')
            #     # 使用savefig保存图像。这里指定文件名和分辨率
            #     plt.savefig('loss_plot.png', dpi=300)

    # 测试环境，评估V2I速率和失败概率
    def test_environment(self):
        V2I_Rate_list = np.zeros(1)
        Fail_percent_list = np.zeros(1)
        test_sample = 200
        Rate_list = []
        time_left_list = []
        # step = 0
        for k in range(test_sample):
            # print(k)
            action_temp = self.action_all_with_power.copy()
            # start = time.perf_counter()
            for i in range(len(self.env.vehicles)):
                # 将所有车辆的信道选择赋值为-1，代表暂未分配动作（初始化）
                self.action_all_with_power[i, :, 0] = -1
                # 根据individual_time_limit对车辆索引进行排序，优先处理通信时间最少的车辆
                sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                #因为j是排序后的索引，所以j只能是（0，1，2）中的一个值，（可以确保3i+j是idx中的唯一索引）
                for j in sorted_idx:
                    # state_old = self.get_state([i,j])
                    # step = step + 1
                    idx = []
                    #i是车辆的索引，j是车辆的目的地索引
                    idx.append(3 * i + j)
                    #得到当前发信收信车辆之间链路状态
                    state_old = self.get_state([i, j])
                    #仅选择前六十个特征(猜测是V2I_channel, V2V_interference, V2V_channel)
                    self.G.features[3 * i + j, :] = state_old[:60]
                    #该状态通过GraphSAGE模型生成嵌入，有助与更好的决策
                    node_embeddings = self.G.use_GraphSAGE(self.channel_reward, 0, idx,
                                                           self.GraphSAGE)
                    max_value = np.max(node_embeddings)
                    # 计算比例因子
                    scale_factor = max_value
                    # 将数组中的每个值除以比例因子（归一化）
                    node_embeddings = node_embeddings / (scale_factor + 0.0001)
                    # 从node_embeddings数组中删除任何不必要的维度，特别是如果它的形状像(1, n)或(n, 1)
                    node_embeddings = np.squeeze(node_embeddings)
                    # print('node_embeddings',node_embeddings.shape)
                    # print('state_old[3*i+j]', state_old.shape)
                    # 创建了一个增强的状态表示，其中包含了原始状态以及从GraphSAGE模型中学习到的新特征（特征融合）
                    better_state_old = np.concatenate((node_embeddings, state_old), axis=0)
                    action = self.predict(better_state_old, 0, True)
                    self.merge_action([i, j], action)
                # mid_time_1 = time.perf_counter()
                #每vehicles/10次 更新一次reward和percent
                if i % (len(self.env.vehicles) // 10) == 1:
                    action_temp = self.action_all_with_power.copy()
                    # print("actions_temp", action_temp)
                    reward, percent = self.env.act_asyn(action_temp)  # self.action_all)
                    # print('reward', np.sum(reward))
                    Rate_list.append(np.sum(reward))

        V2I_Rate_list[0] = np.mean(np.asarray(Rate_list))

        mean_of_V2I_Rate = np.mean(V2I_Rate_list[0])
        print('Mean of the V2I rate is that ', mean_of_V2I_Rate)
        print('Mean of Fail percent is that ', percent)
        return mean_of_V2I_Rate, percent
        # print('action is that', action_temp[0,:])
    # print('Test Reward is ', np.mean(test_result))
            
    def q_learning_mini_batch(self):

        # Training the DQN model
        # ------ 
        #s_t, action,reward, s_t_plus_1, terminal = self.memory.sample() 
        s_t, s_t_plus_1, action, reward = self.memory.sample()  #从缓冲区中采样
        #print() 
        #print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])        
        t = time.time()

        if self.double_q:       #double Q learning   
            pred_q = self.dqn.forward(s_t_plus_1)
            action_tensor = tf.argmax(pred_q, axis=1)
            pred_action = action_tensor.numpy()
            self.target_q = self.dqn.forward_target(s_t_plus_1)
            self.target_q_idx = [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
            q_t_plus_1_with_pred_action = self.target_q_with_idx.numpy()
            target_q_t = self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.dqn.forward_target(s_t_plus_1)
            max_q_t_plus_1 = tf.reduce_max(q_t_plus_1, axis=1)
            max_q_t_plus_1 = max_q_t_plus_1.numpy()
            target_q_t = self.discount * max_q_t_plus_1 + reward
        #_, q_t, loss, w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t, self.action:action, self.s_t:s_t, self.learning_rate_step: self.step}) # training the network
        target_q_t = tf.cast(target_q_t, tf.float32)
        loss, q_t = self.dqn.train_step(s_t, target_q_t, action)
        q_t = q_t.numpy()
        loss = loss.numpy()
        print('loss is ', loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def play(self, n_step = 100, n_episode = 100, test_ep = None, render = False):
        number_of_game = 100
        V2I_Rate_list = np.zeros(number_of_game)
        self.training = False
        self.GraphSAGE = False
        Fail_percent_list = np.zeros(number_of_game)
        self.dqn.model.load_weights('weight/dqn_weights.h5')
        print('dqn_model_weights_load')
        self.env.new_random_game(self.num_vehicle)
        better_state = self.initial_better_state(0, self.GraphSAGE)
        self.G.G_model.load_weights('weight/GNN_weights.h5')
        print('GNN_model_weights_load')

        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            better_state = self.initial_better_state(0, self.GraphSAGE)
            test_sample = 200
            Rate_list = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []
            # step = 0
            for k in range(test_sample):
                #print(k)
                action_temp = self.action_all_with_power.copy()
                # start = time.perf_counter()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        # state_old = self.get_state([i,j])
                        # step = step + 1
                        idx = []
                        idx.append(3 * i + j)
                        state_old = self.get_state([i, j])
                        self.G.features[3 * i + j, :] = state_old[:60]
                        node_embeddings = self.G.use_GraphSAGE(self.channel_reward,0, idx,
                                                               self.GraphSAGE)
                        max_value = np.max(node_embeddings)
                        # 计算比例因子
                        scale_factor = max_value
                        # 将数组中的每个值除以比例因子
                        node_embeddings = node_embeddings / (scale_factor + 0.0001)
                        node_embeddings = np.squeeze(node_embeddings)
                        # print('node_embeddings',node_embeddings.shape)
                        # print('state_old[3*i+j]', state_old.shape)
                        better_state_old = np.concatenate((node_embeddings, state_old), axis=0)
                        action = self.predict(better_state_old, 0, True)
                        self.merge_action([i, j], action)

                        if state_old[-1] <= 0:
                            continue
                        power_selection = int(np.floor(action/self.RB_number))
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])
                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                    # mid_time_1 = time.perf_counter()
                    if i % (len(self.env.vehicles) // 5) == 1:
                        action_temp = self.action_all_with_power.copy()
                    #print("actions_temp", action_temp)
                        reward, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        print('reward', np.sum(reward))
                        Rate_list.append(np.sum(reward))
                    # mid_time_2 = time.perf_counter()
                    # elapsed_time_1 = mid_time_1 - start
                    # elapsed_time_2 = mid_time_2 - mid_time_1
                    # print(f"第一部分代码运行时间：{elapsed_time_1:.8f}秒")
                    # print(f"第二部分代码运行时间：{elapsed_time_2:.8f}秒")
                #print('Rate_list', Rate_list)
                # print("actions", self.action_all_with_power)
                #print(k)
                # print('action', self.action_all_with_power.shape)
                # channel_choices = self.action_all_with_power[:, :, 0].astype(int).flatten()
                #
                # # 统计每个信道被选择的次数
                # channel_counts = np.bincount(channel_choices, minlength=20)
                #
                # # 转换为列表（可选）
                # channel_counts = channel_counts.tolist()
                #
                # # 打印结果
                # for channel_index, count in enumerate(channel_counts):
                #     print(f"信道 {channel_index} 被选择了 {count} 次")

            number_0, bin_edges = np.histogram(power_select_list_0, bins= 10)

            number_1, bin_edges = np.histogram(power_select_list_1, bins= 10)

            number_2, bin_edges = np.histogram(power_select_list_2, bins= 10)


            p_0 = number_0 / (number_0 + number_1 + number_2)
            p_1 = number_1 / (number_0 + number_1 + number_2)
            p_2 = number_2 / (number_0 + number_1 + number_2)

            plt.figure()
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 23 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 10 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 5 dB')
            plt.xlim([0, 0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.savefig('GNN-DDQN')
            #plt.show()

            y_0 = number_0 + number_1 + number_2
            # 画出y_0的直方图
            plt.figure()
            # 计算条形的宽度，这里假设所有的bins都有相同的宽度
            bar_width = (bin_edges[1] - bin_edges[0]) * 0.1
            # 绘制直方图，条形的中心需要对齐到bin_edges，所以需要偏移宽度的一半
            plt.bar(bin_edges[:-1] * 0.1 + 0.01, y_0, width=bar_width, color='c', label='Total Power Selections')
            plt.xlim([0, 0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Vehicle Distribution")
            plt.legend()
            plt.grid()
            # plt.show()  # 用于显示图形
            plt.savefig('GNN-DDQN_y0.png')
            
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            Fail_percent_list[game_idx] = percent
            if game_idx > 0:
                mean_of_V2I_Rate = np.mean(V2I_Rate_list[0:game_idx+1])
                mean_Fail_percent= np.mean(Fail_percent_list[0:game_idx+1])
            else:
                mean_of_V2I_Rate = np.mean(V2I_Rate_list[0])
                mean_Fail_percent = np.mean(Fail_percent_list[0])

            print('Mean of the V2I rate is that ', mean_of_V2I_Rate)
            print('Mean of Fail percent is that ', percent, mean_Fail_percent)
            # print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        # print('Test Reward is ', np.mean(test_result))

    def play_complete_graph(self, n_step=100, n_episode=100, test_ep=None, render=False):
        number_of_game = 100
        V2I_Rate_list = np.zeros(number_of_game)
        self.training = False
        self.GraphSAGE = False
        Fail_percent_list = np.zeros(number_of_game)
        self.dqn.model.load_weights('weight/dqn_weights.h5')
        print('dqn_model_weights_load')
        self.env.new_random_game(self.num_vehicle)
        better_state = self.initial_better_state(0, self.GraphSAGE)
        self.G.G_model.load_weights('weight/GNN_weights.h5')
        print('GNN_model_weights_load')

        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            better_state = self.initial_better_state(0, self.GraphSAGE)
            test_sample = 200
            Rate_list = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []
            for k in range(test_sample):
                # print(k)
                action_temp = self.action_all_with_power.copy()
                # start = time.perf_counter()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        # state_old = self.get_state([i,j])
                        idx = []
                        idx.append(3 * i + j)
                        state_old = self.get_state([i, j])
                        self.G.features[3 * i + j, :] = state_old[:60]
                        start = time.perf_counter()
                        node_embeddings = self.G.test_Complete(self.channel_reward, 0, idx,
                                                               self.GraphSAGE)
                        mid_time_1 = time.perf_counter()
                        max_value = np.max(node_embeddings)
                        # 计算比例因子
                        scale_factor = max_value
                        # 将数组中的每个值除以比例因子
                        node_embeddings = node_embeddings / (scale_factor + 0.0001)
                        node_embeddings = np.squeeze(node_embeddings)
                        # print('node_embeddings',node_embeddings.shape)
                        # print('state_old[3*i+j]', state_old.shape)
                        better_state_old = np.concatenate((node_embeddings, state_old), axis=0)
                        action = self.predict(better_state_old, 0, True)
                        self.merge_action([i, j], action)

                        elapsed_time_1 = mid_time_1 - start
                        # print(f"完全图代码一次运行时间：{elapsed_time_1:.8f}秒")
                        time_left_list.append(elapsed_time_1)

                if time_left_list:
                    average_time = sum(time_left_list) / len(time_left_list)
                    print(f"所有循环的平均运行时间：{average_time:.8f}秒")
                else:
                    print("没有记录到任何运行时间。")
    def play_un_complete_graph(self, n_step=100, n_episode=100, test_ep=None, render=False):
        number_of_game = 100
        V2I_Rate_list = np.zeros(number_of_game)
        self.training = False
        self.GraphSAGE = False
        Fail_percent_list = np.zeros(number_of_game)
        self.dqn.model.load_weights('weight/dqn_weights.h5')
        print('dqn_model_weights_load')
        self.env.new_random_game(self.num_vehicle)
        better_state = self.initial_better_state(0, self.GraphSAGE)
        self.G.G_model.load_weights('weight/GNN_weights.h5')
        print('GNN_model_weights_load')

        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            better_state = self.initial_better_state(0, self.GraphSAGE)
            test_sample = 200
            Rate_list = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []
            for k in range(test_sample):
                # print(k)
                action_temp = self.action_all_with_power.copy()
                # start = time.perf_counter()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        # state_old = self.get_state([i,j])
                        # start = time.perf_counter()
                        idx = []
                        idx.append(3 * i + j)
                        state_old = self.get_state([i, j])
                        self.G.features[3 * i + j, :] = state_old[:60]
                        first_order_neighs, second_order_neighs = self.G.test_get_neighs(idx)
                        start = time.perf_counter()
                        node_embeddings = self.G.test_unComplete(idx,first_order_neighs, second_order_neighs)
                        mid_time_1 = time.perf_counter()
                        # mid_time_1 = time.perf_counter()
                        max_value = np.max(node_embeddings)
                        # 计算比例因子
                        scale_factor = max_value
                        # 将数组中的每个值除以比例因子
                        node_embeddings = node_embeddings / (scale_factor + 0.0001)
                        node_embeddings = np.squeeze(node_embeddings)
                        # print('node_embeddings',node_embeddings.shape)
                        # print('state_old[3*i+j]', state_old.shape)
                        better_state_old = np.concatenate((node_embeddings, state_old), axis=0)
                        action = self.predict(better_state_old, 0, True)

                        self.merge_action([i, j], action)

                        elapsed_time_1 = mid_time_1 - start
                        # print(f"非完全图代码一次运行时间：{elapsed_time_1:.8f}秒")
                        time_left_list.append(elapsed_time_1)

                if time_left_list:
                    average_time = sum(time_left_list) / len(time_left_list)
                    print(f"所有循环的平均运行时间：{average_time:.8f}秒")
                else:
                    print("没有记录到任何运行时间。")

def main():
    up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
    down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
                  750 - 3.5 / 2]
    left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]
    right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                   1299 - 3.5 / 2]
    width = 750
    height = 1299
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
    Env.new_random_game()

    # TensorFlow 2.x 适配
    # 如果需要使用特定的 GPU 或进行内存分配等配置，可以使用 tf.config
    # tf.config.experimental.set_virtual_device_configuration(...)
    # tf.config.gpu.set_per_process_memory_fraction(...)
    # tf.config.gpu.set_allow_growth(True)
    agent = Agent([], Env)  # 注意，在 TensorFlow 2.x 中，不再需要传递会话(sess)
    # agent.train()
    # agent.play()


if __name__ == '__main__':
    main()




