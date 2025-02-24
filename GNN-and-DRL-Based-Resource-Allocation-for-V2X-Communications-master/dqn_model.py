import tensorflow as tf


class DQNModel:
    def __init__(self):
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0005 #最小学习率
        self.learning_rate_decay = 0.96 #学习率衰减率
        self.learning_rate_decay_step = 500000  #学习率衰减步数
        self.initialize_network()
        self.compile_model()

    def initialize_network(self):
        self.model = self.build_dqn()
        self.target_model = self.build_dqn()

    #模型结构，输入102维，输出60维
    def build_dqn(self):
        n_input = 102   #基础状态向量82维（60维信道信息（V2I、V2V干扰、V2V信道）20维邻居选择信息NeiSelection 1维time_remaining和1维load_remaining）+ GraphSAGE嵌入20维（node_embeddings）
        n_output = 60   #输出的动作 （20个资源块*3个功率等级）
        #连接DQN网络结构（102 -> 700 -> 350 -> 180 -> 60），使用ReLU激活函数
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_input,)),
            tf.keras.layers.Dense(700, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
            tf.keras.layers.Dense(350, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
            tf.keras.layers.Dense(180, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
            tf.keras.layers.Dense(n_output, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        ])

        return model

    def forward(self, inputs):
        # 使用主网络进行前向传播，
        # 启用训练模式
        inputs = tf.reshape(inputs, [-1, 102])  #重塑输入维度为(batch_size, 102)
        return self.model(inputs, training=True)

    def forward_target(self, inputs):
        # 使用目标网络进行前向传播，
        # 禁用训练模式
        return self.target_model(inputs, training=False)

    #实现目标网络的软更新，将主网络的权重复制到目标网络
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    
    #实现Huber损失函数
    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    
    #实现指数衰减的学习率调度
    def compile_model(self):
        # 设置学习率衰减
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.learning_rate_decay_step,
            decay_rate=self.learning_rate_decay,
            staircase=True #梯度式衰减
        )

        # 使用闭包来确保能够在优化器中使用动态学习率
        def minimum_lr_fn():
            step_lr = lr_schedule(self.model.optimizer.iterations)
            return tf.maximum(step_lr, self.learning_rate_minimum)
        # 配置RMSprop优化器
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=minimum_lr_fn, rho=0.95, epsilon=0.01)
        # 编译模型
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')  # 使用MSE作为损失函数
        # self.model.build(input_shape=(None, 102))
        # self.model.load_weights('weight/dqn_weights.h5')

    @tf.function
    def train_step(self, inputs, targets, actions):
        with tf.GradientTape() as tape: # 自动求导
            q_values = self.forward(inputs)     # 前向传播， 获取当前状态的Q值
            action_masks = tf.one_hot(actions, q_values.shape[1])   # 将动作转换为one-hot编码
            q_acted = tf.reduce_sum(q_values * action_masks, axis=1)    # 获取选中动作的Q值
            loss = tf.reduce_mean(tf.square(targets - q_acted))     # MSE损失

        grads = tape.gradient(loss, self.model.trainable_variables)     # 计算梯度
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))    # 应用梯度更新
        return loss, q_values   # 返回损失和Q值
