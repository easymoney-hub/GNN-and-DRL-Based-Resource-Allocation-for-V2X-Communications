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

    #更新V2V信道的快衰弱，同样使用瑞利衰弱模型
    def update_fast_fading(self):
        # 维度是(self.n_Veh, self.n_Veh, self.n_RB)
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

    #计算V2I信道的快衰落， 模拟的是Rayleigh 衰落，并且以 dB 为单位输出快衰弱值
    def update_fast_fading(self):
        # 维度是(self.n_Veh, self.n_RB)
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
        self.sig2_dB = -114 #描述接收器中的本底噪声。例如，在 1 MHz 带宽中，热噪声大约为 - 114 dBm
        self.bsAntGain = 8  # 基站天线增益
        self.bsNoiseFigure = 5 # 基站噪声系数
        self.vehAntGain = 3   # 车辆天线增益
        self.vehNoiseFigure = 9  # 车辆噪声系数
        self.sig2 = 10**(self.sig2_dB/10)  # 热噪声,根据dB值计算得到的实际信号功率
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

    #更新一次所有车辆的位置（所有车辆前进单位距离--delta_distance）
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
            #方向为上
            if self.vehicles[i].direction == 'u':
                #print ('len of position', len(self.position), i)
                #判断是否到达左拐路口
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] <=self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):
                        #40%概率变道
                        if (random.uniform(0,1) < 0.4):
                            #转向后的横坐标=转向前的横坐标-（单位时间路程-离路口的距离）
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),self.left_lanes[j] ] 
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                # 如果没有向左变道，判断是否到达右拐路口（操作与左拐类似）
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <=self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j] ] 
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                #如果没有变道，直接前进
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            #方向为下
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
            #方向为右
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
            #方向为左
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
            # 边界检查（if it comes to an exit）如果车辆超出定义的环境边界（宽度或高度），更改方向（顺时针改变），重新定位到地图的另一侧
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

    # 根据车辆当前的位置更新每辆车的邻居信息
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
        # 计算车辆间的距离，位置矩阵转置后减去原矩阵，再求模，得到车辆间距离
        Distance = abs(z.T-z)
        for i in range(len(self.vehicles)):
            #对第i列进行排序，按距离，返回排序后的索引
            sort_idx = np.argsort(Distance[:,i])
            #选择三辆车作为邻居
            for j in range(3):
                self.vehicles[i].neighbors.append(sort_idx[j+1])
            # 用于从排序后的索引数组中随机选择三个不重复的索引作为目的地
            # ort_idx[1:int(len(sort_idx)/5) 从sort_idx切片，中选择3个数，不重复
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
        """这个固定增益是怎么定义的还没搞清楚"""
        # 信道绝对衰减值：将路损和Shadow衰减相加，再加一个固定增益，得到V2V信道的绝对衰减值;V2I信道无固定增益
        # 增益：创建一个大小为len(self.vehicles)*len(self.vehicles)的单位矩阵，单位矩阵是一个主对角线元素都为1，其余元素都为0的方阵。将这个单位矩阵乘以50
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
        #将V2V_channels_abs由二维扩展到三位并复制n_RB次
        #最终维度(len(self.vehicles)，len(self.vehicles), self.n_RB)
        # 将同一组路径损耗和阴影衰落值，扩展到每个资源块。这样做的目的是确保每个资源块的信道增益都能得到更新和调整，使得每个资源块都能根据对应的快衰弱进行修正
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        # 当快衰弱从绝对信道值中减去时，表示对由于快衰弱引起的瞬时波动进行调整，以得到更稳定的信道增益。
        # 可以观察到的平均信号强度，提供了一个稳定的信号强度，以更好地评估系统的长期性能，确保短期波动不会扭曲整体结果。
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - self.V2Vchannels.FastFading
        #对V2I信道操作道理同上
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

    def Compute_Performance_Reward_Batch(self, actions_power, idx):    # action with prwer selection
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        #提取信道选择部分
        actions = actions_power.copy()[:, :, 0]           #
        #提取功率选择部分
        power_selection = actions_power.copy()[:,:,1]   #
        # print(actions)
        origin_channel_selection = actions[idx[0], idx[1]]

        """计算V2V信道中所有车辆对的信号强度和所受干扰"""
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        Interfence_times = np.zeros((len(self.vehicles), 3))    # 3 neighbors
        actions[idx[0], idx[1]] = 100  # something not relavant
        for i in range(self.n_RB):
            # 找出数组actions中所有等于i的元素的索引 （action中所有选择第i个资源块信道的发收车辆对的索引）
            indexes = np.argwhere(actions == i)
            #print('index',indexes)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                # indexes[j, 0]是发送车辆的索引，indexes[j, 1]是接收车辆的索引
                # 执行与资源块i相关动作的第j辆车辆的索引
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                #当前发收方之间的信号强度
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j,0], receiver_j, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                #当前发收方之间的干扰强度
                V2V_Interference[indexes[j,0],indexes[j,1]] +=  10**((self.V2I_power_dB- self.V2V_channels_with_fastfading[i,receiver_j,i] + \
                2*self.vehAntGain - self.vehNoiseFigure)/10)  # interference from the V2I links
                
                for k in range(j+1, len(indexes)):   #当前信道中其余V2V通信对对当前考虑的V2V的干扰
                    receiver_k = self.vehicles[indexes[k,0]].destinations[indexes[k,1]]
                    #累加使用同信道的其他车辆对当前车辆的干扰
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[k,0],receiver_j,i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    #累加当前车辆的信号对使用同信道的其他车辆的干扰
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j,0], receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    #这两行同时存在，那Interfence_times代表的是一对链路作为干扰源和被干扰方的总次数，如果只想要被干扰的次数，应该删去第二行
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1
        #对计算出来的V2V信道干扰总体加上一个热噪声，赋给类对象记录
        self.V2V_Interference = V2V_Interference + self.sig2
        

        # 初始化V2V速率列表和需求缺口列表
        V2V_Rate_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))  # the number of RB times the power level
        Deficit_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        for i in range(self.n_RB):#遍历每个资源块
            indexes = np.argwhere(actions == i)
            V2V_Signal_temp = V2V_Signal.copy()            
            #receiver_k = self.vehicles[idx[0]].neighbors[idx[1]]
            receiver_k = self.vehicles[idx[0]].destinations[idx[1]] 
            for power_idx in range(len(self.V2V_power_dB_List)): #遍历每个功率等级
                V2V_Interference_temp = V2V_Interference.copy()
                # 计算当前信道和功率选择下的V2V信号强度
                 #四项分别是  V2V发送功率(dB)-信道衰落-噪声系数+车辆天线增益(2倍是因为发送和接收都有增益)
                V2V_Signal_temp[idx[0],idx[1]] = 10**((self.V2V_power_dB_List[power_idx] - \
                self.V2V_channels_with_fastfading[idx[0], self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)
                # 计算当前信道和功率选择下的V2V干扰强度
                # 四项分别是  V2I发送功率(dB)-信道衰落-噪声系数+车辆天线增益(2倍是因为发送和接收都有增益)
                V2V_Interference_temp[idx[0],idx[1]] +=  10**((self.V2I_power_dB - \
                self.V2V_channels_with_fastfading[i,self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                #遍历每个使用同一个信道的用户中
                for j in range(len(indexes)):
                    receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                    #再增加其他车辆通信对当前车辆通信的干扰
                    V2V_Interference_temp[idx[0],idx[1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0], indexes[j,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    #计算当前车辆通信对其他车辆通信的干扰
                    V2V_Interference_temp[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_idx]-\
                    self.V2V_channels_with_fastfading[idx[0],receiver_j, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                # 计算香农公式下的传输速率
                V2V_Rate_cur = np.log2(1 + np.divide(V2V_Signal_temp, V2V_Interference_temp))
                # 当评估到的信道和功率组合与实际使用的相同时，保存当前的传输速率
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    V2V_Rate = V2V_Rate_cur.copy()
                # 二维数组，记录每个信道(i)和功率等级(power_idx)组合下的总传输速率，将所有链路的速率相加得到总速率
                V2V_Rate_list[i, power_idx] = np.sum(V2V_Rate_cur)
                #  self.demand：剩余需要传输的数据量  V2V_Rate_cur * 1500: 当前速率下1s可传输的数据量   self.individual_time_limit:剩余的传输时间限制
                # 如果为正，表示在剩余时间内无法传完所需数据；反之则可以传完
                Deficit_list[i,power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(V2V_Signal_temp.shape), (self.demand - self.individual_time_limit * V2V_Rate_cur * 1500)))
        
        """"计算V2I的速率和干扰"""
        Interference = np.zeros(self.n_RB)  
        #V2I下不同子信道在不同功率下的发送速率
        V2I_Rate_list = np.zeros((self.n_RB,len(self.V2V_power_dB_List)))    # 3 of power level
        #遍历所有车辆
        for i in range(len(self.vehicles)):
            # 遍历第i辆车的所有通信对
            for j in range(len(actions[i,:])):
                #跳过当前考虑的车辆对
                if (i ==idx[0] and j == idx[1]):
                    continue
                #计算每个V2V链路对基站的干扰
                #五项分别是:V2V发送功率(dB)-信道衰落+车辆天线增益(不是2倍,一辆车)+基站天线增益-基站噪声系数
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] - \
                self.V2I_channels_with_fastfading[i, actions[i][j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10) 
        V2I_Interference = Interference + self.sig2 #加入热噪声
        
        for i in range(self.n_RB):            
            for j in range(len(self.V2V_power_dB_List)):
                V2I_Interference_temp = V2I_Interference.copy()
                #加入当前考虑的V2V链路对基站的干扰
                # 五项分别是:V2V发送功率(dB)-信道衰落+车辆天线增益(不是2倍,一辆车)+基站天线增益-基站噪声系数
                V2I_Interference_temp[i] += 10**((self.V2V_power_dB_List[j] - self.V2I_channels_with_fastfading[idx[0], i] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
                # 计算当前子信道下当前功率时 V2I传输速率（香农公式） 
                # [0:min(self.n_RB,self.n_Veh)]  每个资源块最多只能分配给一辆车辆进行V2I通信,每辆车最多只能使用一个资源块进行V2I通信;如果有10个资源块但只有8辆车，那么最多只能使用8个资源块进行V2I通信;如果有8个资源块但有10辆车，那么最多只能使用8个资源块进行V2I通信
                V2I_Rate_list[i, j] = np.sum(np.log2(1 + np.divide(10**((self.V2I_power_dB + self.vehAntGain + self.bsAntGain \
                - self.bsNoiseFigure-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)])/10), V2I_Interference_temp[0:min(self.n_RB,self.n_Veh)])))
        
        #更新剩余需求量--从剩余需求量中减去这个时间段内传输的数据量
        self.demand -= V2V_Rate * self.update_time_train * 1500
        # 更新时间计数器,减去已用时间
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train
        
        if self.demand[idx[0], idx[1]] < 0: # 如果当前车辆对的需求已满足
            time_left = self.V2V_limit  # 重置为最大时间限制
        else:
            time_left = self.individual_time_limit[idx[0], idx[1]]  # 否则使用剩余时间限制
        # 车辆V2V消息发送时延超出限制或提前发送完毕时重置时延，重新发送或发送新的内容
        self.individual_time_limit[np.add(self.individual_time_limit <= 0,  self.demand < 0)] = self.V2V_limit   
        self.demand[self.demand < 0] = self.demand_amount
        #重置测试时间计数：
        if self.test_time_count == 0: 
            self.test_time_count = 10
        # print('time_left', time_left)
        return V2I_Rate_list, Deficit_list, time_left

    # 计算在V2V中，其他车辆对当前车辆通信通道的干扰。该方法考虑了来自车辆到车辆（V2V）和车辆到基础设施（V2I）通道的干扰
    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        V2V_Interference = np.zeros((len(self.vehicles), 3, self.n_RB)) + self.sig2
        if len(actions.shape) == 3:
            channel_selection = actions.copy()[:,:,0]
            power_selection = actions[:,:,1]
            channel_selection[np.logical_not(self.activate_links)] = -1  #将未激活的链路设置为 -1
            """（V2I）通道的干扰"""
            for i in range(self.n_RB):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k,:])): #遍历车辆所有链路
                        #[k,m]车辆对在信道i下的收到的干扰:  V2I发送功率 - 信道衰弱 -+两辆车天线增益 -噪声系数
                        V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + \
                        2 * self.vehAntGain - self.vehNoiseFigure)/10)
            """（V2V）通道的干扰"""
            #干扰源--所有非当前链路的已激活链路
            for i in range(len(self.vehicles)):
                for j in range(len(channel_selection[i,:])):
                    #收干扰链路
                    for k in range(len(self.vehicles)):
                        for m in range(len(channel_selection[k,:])):
                            if (i==k) or (channel_selection[i,j] >= 0): # 跳过同一车辆 && 跳过未激活的链路(-1表示未激活)
                                continue
                            #表示车辆k与其第m个目标在特定信道上受到的干扰功率（累加--所有的其他的通信链路对当前链路所造成的累计干扰）
                            # [k]：接收车辆的索引  [m]：接收车辆的第m个通信目标  [channel_selection[i,j]]：干扰源使用的信道编号
                            V2V_Interference[k, m, channel_selection[i,j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] -\
                            self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2*self.vehAntGain - self.vehNoiseFigure)/10)

        #最后返回的结果是V2I和V2V干扰的和
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)
                
        
    def renew_demand(self):
        # generate a new demand of a V2V
        self.demand = self.demand_amount*np.ones((self.n_RB,3))
        self.time_limit = 10

    #根据传入的动作，更新位置，信道快衰弱，计算干扰 --返回综合计算后的奖励
    def act_for_training(self, actions, idx):
        # =============================================
        # This function gives rewards for training
        # ===========================================
        rewards_list = np.zeros(self.n_RB)
        action_temp = actions.copy()
        self.activate_links = np.ones((self.n_Veh,3), dtype = 'bool')  # 激活所有链路
        #赋给三个数组的值分别是 V2I_Rate_list, Deficit_list, time_left
        V2I_rewardlist, V2V_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp,idx)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions) 
        #重塑奖励列表为一维数组---reshape([-1])会将数组展平成一维数组
        rewards_list = rewards_list.T.reshape([-1])
        V2I_rewardlist = V2I_rewardlist.T.reshape([-1])
        V2V_rewardlist = V2V_rewardlist.T.reshape([-1])
        # 归一化V2I和V2V奖励---(value - min_value) / (max_value - min_value)---值会线性映射到[0,1]区间
        # value_idx = channel_idx + 20*power_idx   在一维数组中的索引 --20是因为信道数为20
        # V2I_rewardlist[动作中的车辆对在重塑的一维数组中的索引] / {np.max(V2I_rewardlist) - np.min(V2I_rewardlist) + 0.000001} 
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
        t = lambdda * V2I_reward + (1 - lambdda) * V2V_reward   # 计算综合奖励--V2I奖励权重为0.1 , V2V奖励权重为0.9
        # print("time left", time_left) 
        # print ("Reward", V2I_reward, V2V_reward, time_left)
        #return t
        return t - (self.V2V_limit - time_left)/self.V2V_limit  # 返回最终奖励，考虑时间惩罚
        
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
    
    #起个新环境--部署车辆，初始化信道，初始化时间计数，初始化各种记录列表
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
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2  #self.sig2可能模拟的是背景噪声
        self.demand_amount = 30 #需求数
        self.demand = self.demand_amount * np.ones((self.n_Veh,3))
        self.test_time_count = 10 #测试的时间计数
        self.V2V_limit = 0.1  # 100 ms--V2V toleratable latency
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh,3)) #每个车辆通信会话时间限制
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh,3)) #每个车辆通信会话之间的时间间隔
        self.UnsuccessfulLink = np.zeros((self.n_Veh,3)) #记录不成功的链路
        self.success_transmission = 0 #记录成功传输的次数
        self.failed_transmission = 0 #记录失败传输的次数
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.002 # 2ms update time for testing
        self.update_time_asyn = 0.0002 # 0.2 ms update one subset of the vehicles; for each vehicle, the update time is 2 ms
        self.activate_links = np.zeros((self.n_Veh,3), dtype='bool') #记录激活的链路，默认值是False

if __name__ == "__main__":
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height) 
    Env.test_channel()    
