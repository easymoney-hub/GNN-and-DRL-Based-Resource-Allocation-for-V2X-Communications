from __future__ import division
import numpy as np
import time
import random
import math
# This file is revised for more precise and concise expression.
class V2Vchannels:              
    # Simulator of the V2V Channels
    def __init__(self, n_Veh, n_RB):
        self.t = 0
        self.h_bs = 1.5# 基站高度 因为是在V2V场景中，所以bs和ms的高度相同
        self.h_ms = 1.5# 移动台高度
        self.fc = 2# 载波频率
        self.decorrelation_distance = 10# 去相关距离，用于描述信号相关性随距离变化的一个参数
        self.shadow_std = 3 #阴影衰落的标准差，用于描述无线信号在传播过程中由于阴影效应导致的信号强度波动的统计特性。
        self.n_Veh = n_Veh# 车辆数量
        self.n_RB = n_RB# 资源块数量
        self.update_shadow([])
    def update_positions(self, positions):
        self.positions = positions

    # 更新V2V信道的路径损耗
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    # 根据delta_distance更新阴影衰减效果。 V2V中的self.Shadow是二维的
    def update_shadow(self, delta_distance_list):
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        #初始化时delta_diatance为空，确保了之后delta-distance赋值后，更新shadow时self.Shadow不为空，且是服从均值为0，标准差为self.shadow_std的正态分布
        if len(delta_distance_list) == 0:
            self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_Veh))
        else:
            self.Shadow = np.exp(-1*(delta_distance/self.decorrelation_distance)) * self.Shadow +\
                         np.sqrt(1 - np.exp(-2*(delta_distance/self.decorrelation_distance))) * np.random.normal(0, self.shadow_std, size = (self.n_Veh, self.n_Veh))

    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB) ) + 1j * np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))

    #计算并返回两个车辆位置之间的路径损耗。
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001 #+0.001：避免距离d为零，防止后续计算中出现除以零的错误
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)
        # 根据距离d计算视线（LOS）路径损耗
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        #根据两个距离参数d1和d2计算非视线（NLOS）路径损耗。
        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)

        # 判断是否为视线（LOS）情况，并计算路径损耗，更改当前vehical对象的ifLos属性和shadow_std属性
        if min(d1,d2) < 7:
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4                      # if Non line of sight, the std is 4
        return PL

class V2Ichannels: 
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_RB):
        self.h_bs = 25      # 基站高度
        self.h_ms = 1.5     # 移动台高度
        self.Decorrelation_distance = 50      # 去相关距离，用于描述信号相关性随距离变化的一个参数
        self.BS_position = [750/2, 1299/2]    #假设BS处在画布中央  # Suppose the BS is in the center
        self.shadow_std = 8 #阴影衰落的标准差，用于描述无线信号在传播过程中由于阴影效应导致的信号强度波动的统计特性。
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    #根据传入的车辆位置列表，增加属性：位置
    def update_positions(self, positions):
        self.positions = positions

    # 更新V2I信道的路径损耗
    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1,d2)
            # change from meters to kilometers
            # 128.1 + 37.6*log10(车辆和基站之间的距离)
            self.PathLoss[i] = 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000)

    # 根据delta_distance更新阴影衰减效果；不同于V2V，V2I中的self.Shadow是一维的
    def update_shadow(self, delta_distance_list):
        # 如果传入delta_distance为空,则self.Shadow 服从均值为 0、标准差为self.shadow_std 的正态分布
        if len(delta_distance_list) == 0:  # initialization
            self.Shadow = np.random.normal(0, self.shadow_std, self.n_Veh)
        else: 
            delta_distance = np.asarray(delta_distance_list)
            self.Shadow = np.exp(-1*(delta_distance/self.Decorrelation_distance))* self.Shadow +\
                          np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*np.random.normal(0,self.shadow_std, self.n_Veh)

    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size = (self.n_Veh, self.n_RB)) + 1j* np.random.normal(size = (self.n_Veh, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
class Environ:
    # Enviroment Simulator: Provide states and rewards to agents. 
    # Evolve to new state based on the actions taken by the vehicles.
    def __init__ (self, down_lane, up_lane, left_lane, right_lane, width, height):
        #
        self.timestep = 0.01
        # 四个方向的车辆可选的位置
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        #画布的大小
        self.width = width
        self.height = height
        self.vehicles = []
        self.demands = []  
        self.V2V_power_dB = 23 # dBm
        self.V2I_power_dB = 23 # dBm
        self.V2V_power_dB_List = [23, 10, 5]             # the power levels
        #self.V2V_power = 10**(self.V2V_power_dB)
        #self.V2I_power = 10**(self.V2I_power_dB)
        self.sig2_dB = -114
        self.bsAntGain = 8  # 基站天线增益
        self.bsNoiseFigure = 5 # 基站噪声系数
        self.vehAntGain = 3   # 车辆天线增益
        self.vehNoiseFigure = 9  # 车辆噪声系数
        self.sig2 = 10**(self.sig2_dB/10)  # 根据dB值计算得到的实际信号功率
        self.V2V_Shadowing = [] # V2V通信的阴影衰落列表，用于模拟信号传播中的阴影效应
        self.V2I_Shadowing = [] # V2I通信的阴影衰落列表
        self.delta_distance = [] #deltatime内车辆行驶的距离 delta time slot is 2 ms.
        self.n_RB = 20
        self.n_Veh = 20
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)  # number of vehicles
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))

    def add_new_vehicles(self, start_position, start_direction, start_velocity):    
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    # 根据给定的数量添加新车辆,四个方向上都添加，传入参数需要是总车数/4
    def add_new_vehicles_by_number(self, n):
        for i in range(n):
            ind = np.random.randint(0,len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0,self.height)]
            start_direction = 'd'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            start_position = [self.up_lanes[ind], random.randint(0,self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
        # 二维的正态分布随机数数组，用于模拟车辆间（V2V）通信的阴影衰落 np.random.normal（均值，标准差）
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        # 存储了所有车辆的速度信息
        self.delta_distance = np.asarray([c.velocity for c in self.vehicles])
        #self.renew_channel()
    def renew_positions(self):
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        i = 0
        #for i in range(len(self.position)):
        while(i < len(self.vehicles)):
            #print ('start iteration ', i)
            #print(self.position, len(self.position), self.direction)
            delta_distance = self.vehicles[i].velocity * self.timestep
            change_direction = False
            if self.vehicles[i].direction == 'u':
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    
                    if (self.vehicles[i].position[1] <=self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),self.left_lanes[j] ] 
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <=self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j] ] 
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >=self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - ( self.vehicles[i].position[1]- self.left_lanes[j])), self.left_lanes[j] ] 
                            #print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >=self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + ( self.vehicles[i].position[1]- self.right_lanes[j])),self.right_lanes[j] ] 
                                #print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    
                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance
            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
            # delete
            #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0],self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1],self.vehicles[i].position[1]]
                
            i += 1
    def test_channel(self):
        # ===================================
        #   test the V2I and the V2V channel 
        # ===================================
        self.n_step = 0
        self.vehicles = []
        n_Veh = 20
        self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh/4))
        step = 1000
        time_step = 0.1  # every 0.1s update
        for i in range(step):
            self.renew_positions() 
            positions = [c.position for c in self.vehicles]
            self.update_large_fading(positions, time_step)
            self.update_small_fading()
            print("Time step: ", i)
            print(" ============== V2I ===========")
            print("Path Loss: ", self.V2Ichannels.PathLoss)
            print("Shadow:",  self.V2Ichannels.Shadow)
            print("Fast Fading: ",  self.V2Ichannels.FastFading)
            print(" ============== V2V ===========")
            print("Path Loss: ", self.V2Vchannels.PathLoss[0:3])
            print("Shadow:", self.V2Vchannels.Shadow[0:3])
            print("Fast Fading: ", self.V2Vchannels.FastFading[0:3])

    def update_large_fading(self, positions, time_step):
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = time_step * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
    def update_small_fading(self):
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        
    def renew_neighbor(self):   
        # ==========================================
        # update the neighbors of each vehicle.
        # ===========================================
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
            #print('action and neighbors delete', self.vehicles[i].actions, self.vehicles[i].neighbors)
        #用于存储车辆间的距离
        Distance = np.zeros((len(self.vehicles),len(self.vehicles)))
        # 每辆车的位置信息转换为复数形式
        z = np.array([[complex(c.position[0],c.position[1]) for c in self.vehicles]])
        # 计算车辆间的距离
        Distance = abs(z.T-z)
        for i in range(len(self.vehicles)):
            #对第i列进行排序，按距离，返回距离最近车的索引
            sort_idx = np.argsort(Distance[:,i])
            #选择三辆车作为邻居
            for j in range(3):
                self.vehicles[i].neighbors.append(sort_idx[j+1])                
            destination = np.random.choice(sort_idx[1:int(len(sort_idx)/5)],3, replace = False)
            self.vehicles[i].destinations = destination
        self.Distance = Distance

    # 更新V2V和V2I信道，包括车辆的位置、路损、shadow、
    def renew_channel(self):
        # ===========================================================================
        # This function updates all the channels including V2V and V2I channels
        # =============================================================================
        positions = [c.position for c in self.vehicles]
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = 0.002 * np.asarray([c.velocity for c in self.vehicles])    #delta时间内车辆行驶的距离 time slot is 2 ms.
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
        self.V2V_channels_abs = self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow + 50 * np.identity(
            len(self.vehicles))
        self.V2I_channels_abs = self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow

    # 更新V2V和V2I信道的快衰落
    def renew_channels_fastfading(self):   
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        self.renew_channel()
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - self.V2Vchannels.FastFading
        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - self.V2Ichannels.FastFading
        #print("V2I channels", self.V2I_channels_with_fastfading)

    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):   # revising based on the fast fading part
        actions = actions_power.copy()[:,:,0]  # the channel_selection_part
        power_selection = actions_power.copy()[:,:,1]
        Rate = np.zeros(len(self.vehicles))
        Interference = np.zeros(self.n_RB)  # V2V signal interference to V2I links
        # 遍历所有车辆和动作，如果链路未激活则跳过。计算并累加V2V信号对V2I链路的干扰。
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i,:])):
                if not self.activate_links[i,j]:
                    continue
                #print('power selection,', power_selection[i,j])  
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]]  - self.V2I_channels_with_fastfading[i, actions[i,j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)  # fast fading

        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        
        # remove the effects of none active links
        #print('shapes', actions.shape, self.activate_links.shape)
        #print(not self.activate_links)
        actions[(np.logical_not(self.activate_links))] = -1
        #print('action are', actions)
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                # compute the V2V signal links
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10) 
                if i < self.n_Veh:
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure )/10)  # V2I links interference to V2V links  
                for k in range(j+1, len(indexes)):                  # computer the peer V2V links
                    #receiver_k = self.vehicles[indexes[k][0]].neighbors[indexes[k][1]]
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)               
       
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.zeros(self.activate_links.shape)
        V2V_Rate[self.activate_links] = np.log2(1 + np.divide(V2V_Signal[self.activate_links], self.V2V_Interference[self.activate_links]))

        #print("V2V Rate", V2V_Rate * self.update_time_test * 1500)
        #print ('V2V_Signal is ', np.log(np.mean(V2V_Signal[self.activate_links])))
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)]))


         # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_test * 1500    # decrease the demand
        self.test_time_count -= self.update_time_test               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_test         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_test      # compute the time interval left for next transmission

        # --- update the demand ---
        
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape ) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        #print("demand is", self.demand)
        #print('mean rate of average V2V link is', np.mean(V2V_Rate[self.activate_links]))
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False 
        #print('number of activate links is', np.sum(self.activate_links)) 
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        #if self.n_step % 1000 == 0 :
        #    self.success_transmission = 0
        #    self.failed_transmission = 0
        failed_percentage = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)
        # print('Percentage of failed', np.sum(new_active), self.failed_transmission, self.failed_transmission + self.success_transmission , failed_percentage)    
        return V2I_Rate, failed_percentage #failed_percentage

        
    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):   # revising based on the fast fading part
        # ===================================================
        #  --------- Used for Testing -------
        # ===================================================
        actions = actions_power[:, :, 0]  # the channel_selection_part
        #print('actions', actions)
        #print('self.activate_links', self.activate_links)
        power_selection = actions_power[:, :, 1]
        Interference = np.zeros(self.n_RB)   # Calculate the interference from V2V to V2I
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i, :])):
                if not self.activate_links[i, j]:
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i, j]] - \
                                                     self.V2I_channels_with_fastfading[i, actions[i,j]] + \
                                                     self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)

        #print('Interference', Interference)
        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        Interfence_times = np.zeros((len(self.vehicles), 3))
        actions[(np.logical_not(self.activate_links))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10) 
                if i<self.n_Veh:
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - \
                    self.V2V_channels_with_fastfading[i][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)  # V2I links interference to V2V links
                for k in range(j+1, len(indexes)):
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1               

        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)]))
        #print("V2I information", V2I_Signals, self.V2I_Interference, V2I_Rate)
        
        # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_asyn * 1500    # decrease the demand
        self.test_time_count -= self.update_time_asyn               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_asyn         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_asyn     # compute the time interval left for next transmission

        # --- update the demand ---
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <= 0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        fail_percent = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)            
        return V2I_Rate, fail_percent

    def Compute_Performance_Reward_Batch(self, actions_power, idx):    # add the power dimension to the action selection
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        actions = actions_power.copy()[:, :, 0]           #
        power_selection = actions_power.copy()[:,:,1]   #
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        Interfence_times = np.zeros((len(self.vehicles), 3))    # 3 neighbors
        # print(actions)
        origin_channel_selection = actions[idx[0], idx[1]]
        actions[idx[0], idx[1]] = 100  # something not relavant
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            #print('index',indexes)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j,0], receiver_j, i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                V2V_Interference[indexes[j,0],indexes[j,1]] +=  10**((self.V2I_power_dB- self.V2V_channels_with_fastfading[i,receiver_j,i] + \
                2*self.vehAntGain - self.vehNoiseFigure)/10)  # interference from the V2I links
                
                for k in range(j+1, len(indexes)):   #当前信道中其余V2V通信对对当前考虑的V2V的干扰
                    receiver_k = self.vehicles[indexes[k,0]].destinations[indexes[k,1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[k,0],receiver_j,i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j,0], receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1#这两行同时存在，那Interfence_times代表的是一对链路作为干扰源和被干扰方的总次数，如果只想要被干扰的次数，应该删去第二行
                    
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))  # the number of RB times the power level
        Deficit_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            V2V_Signal_temp = V2V_Signal.copy()            
            #receiver_k = self.vehicles[idx[0]].neighbors[idx[1]]
            receiver_k = self.vehicles[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.V2V_power_dB_List)):
                V2V_Interference_temp = V2V_Interference.copy()
                V2V_Signal_temp[idx[0],idx[1]] = 10**((self.V2V_power_dB_List[power_idx] - \
                self.V2V_channels_with_fastfading[idx[0], self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)
                V2V_Interference_temp[idx[0],idx[1]] +=  10**((self.V2I_power_dB - \
                self.V2V_channels_with_fastfading[i,self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                for j in range(len(indexes)):
                    receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                    V2V_Interference_temp[idx[0],idx[1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0], indexes[j,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference_temp[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_idx]-\
                    self.V2V_channels_with_fastfading[idx[0],receiver_j, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                V2V_Rate_cur = np.log2(1 + np.divide(V2V_Signal_temp, V2V_Interference_temp))
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    V2V_Rate = V2V_Rate_cur.copy()
                V2V_Rate_list[i, power_idx] = np.sum(V2V_Rate_cur)
                Deficit_list[i,power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(V2V_Signal_temp.shape), (self.demand - self.individual_time_limit * V2V_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)  
        V2I_Rate_list = np.zeros((self.n_RB,len(self.V2V_power_dB_List)))    # 3 of power level
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i,:])):
                if (i ==idx[0] and j == idx[1]):
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] - \
                self.V2I_channels_with_fastfading[i, actions[i][j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10) 
        V2I_Interference = Interference + self.sig2
        for i in range(self.n_RB):            
            for j in range(len(self.V2V_power_dB_List)):
                V2I_Interference_temp = V2I_Interference.copy()
                V2I_Interference_temp[i] += 10**((self.V2V_power_dB_List[j] - self.V2I_channels_with_fastfading[idx[0], i] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
                V2I_Rate_list[i, j] = np.sum(np.log2(1 + np.divide(10**((self.V2I_power_dB + self.vehAntGain + self.bsAntGain \
                - self.bsNoiseFigure-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)])/10), V2I_Interference_temp[0:min(self.n_RB,self.n_Veh)])))
                     
        self.demand -= V2V_Rate * self.update_time_train * 1500
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train
        if self.demand[idx[0], idx[1]] < 0:
            time_left = self.V2V_limit
        else:
            time_left = self.individual_time_limit[idx[0], idx[1]]
        self.individual_time_limit[np.add(self.individual_time_limit <= 0,  self.demand < 0)] = self.V2V_limit   # 车辆V2V消息发送时延超出限制或提前发送完毕时重置时延，重新发送或发送新的内容
        self.demand[self.demand < 0] = self.demand_amount
        if self.test_time_count == 0:
            self.test_time_count = 10
        # print('time_left', time_left)
        return V2I_Rate_list, Deficit_list, time_left

    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        V2V_Interference = np.zeros((len(self.vehicles), 3, self.n_RB)) + self.sig2
        if len(actions.shape) == 3:
            channel_selection = actions.copy()[:,:,0]
            power_selection = actions[:,:,1]
            channel_selection[np.logical_not(self.activate_links)] = -1
            for i in range(self.n_RB):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k,:])):
                        V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + \
                        2 * self.vehAntGain - self.vehNoiseFigure)/10)
            for i in range(len(self.vehicles)):
                for j in range(len(channel_selection[i,:])):
                    for k in range(len(self.vehicles)):
                        for m in range(len(channel_selection[k,:])):
                            if (i==k) or (channel_selection[i,j] >= 0):
                                continue
                            V2V_Interference[k, m, channel_selection[i,j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] -\
                            self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2*self.vehAntGain - self.vehNoiseFigure)/10)

        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)
                
        
    def renew_demand(self):
        # generate a new demand of a V2V
        self.demand = self.demand_amount*np.ones((self.n_RB,3))
        self.time_limit = 10
    def act_for_training(self, actions, idx):
        # =============================================
        # This function gives rewards for training
        # ===========================================
        rewards_list = np.zeros(self.n_RB)
        action_temp = actions.copy()
        self.activate_links = np.ones((self.n_Veh,3), dtype = 'bool')
        V2I_rewardlist, V2V_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp,idx)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions) 
        rewards_list = rewards_list.T.reshape([-1])
        V2I_rewardlist = V2I_rewardlist.T.reshape([-1])
        V2V_rewardlist = V2V_rewardlist.T.reshape([-1])
        V2I_reward = (V2I_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                      np.min(V2I_rewardlist))/(np.max(V2I_rewardlist) -np.min(V2I_rewardlist) + 0.000001)
        V2V_reward = (V2V_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                     np.min(V2V_rewardlist))/(np.max(V2V_rewardlist) -np.min(V2V_rewardlist) + 0.000001)
        lambdda = 0.1
        #print ("Reward", V2I_reward, V2V_reward, time_left)
        # if time_left < 0:
        #     t = V2V_reward
        # else:
        #     t = 3 * time_left * V2I_reward + (1 - 3 * time_left) * V2V_reward
        t = lambdda * V2I_reward + (1 - lambdda) * V2V_reward
        # print("time left", time_left)
        # print ("Reward", V2I_reward, V2V_reward, time_left)
        #return t
        return t - (self.V2V_limit - time_left)/self.V2V_limit
        
    def act_asyn(self, actions):
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()            
            self.renew_channels_fastfading()
        reward = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.Compute_Interference(actions)
        return reward

    def act(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1        
        reward = self.Compute_Performance_Reward_fast_fading_with_power(actions)
        self.renew_positions()            
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward
        
    def new_random_game(self, n_Veh = 0):
        # make a new game
        self.n_step = 0
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))
        self.add_new_vehicles_by_number(int(self.n_Veh/4))#在四个方向上都添加，所以是n_Veh/4
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)  # number of vehicles
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Veh,3))
        self.test_time_count = 10
        self.V2V_limit = 0.1  # 100 ms V2V toleratable latency
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh,3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh,3))
        self.UnsuccessfulLink = np.zeros((self.n_Veh,3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.002 # 2ms update time for testing
        self.update_time_asyn = 0.0002 # 0.2 ms update one subset of the vehicles; for each vehicle, the update time is 2 ms
        self.activate_links = np.zeros((self.n_Veh,3), dtype='bool')

if __name__ == "__main__":
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height) 
    Env.test_channel()    
