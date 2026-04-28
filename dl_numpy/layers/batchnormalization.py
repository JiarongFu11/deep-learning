import numpy as np

class BatchNormalization():
    def __init__(self, num_features:int, momentum:float=0.1):
        self.init_params(num_features, momentum)
    
    def init_params(self, num_features, momentum):
        self.gamma = np.ones((1, num_features, 1, 1))
        self.b = np.zeros((1, num_features, 1, 1))
        
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_std = np.ones((1, num_features, 1, 1))
        self.momentum = momentum
        self.eps = 1e-5
    
    def forward(self, input_data:np.ndarray, train=True) -> np.ndarray:
        
        if input_data.ndim == 4:
            self.axis = (0, 2, 3)
        else:
            self.axis = 0
            if self.gamma.ndim == 4:
                self.gamma = self.gamma.reshape(-1)
                self.beta = self.beta.reshape(-1)
                self.running_mean = self.running_mean.reshape(-1)
                self.running_std = self.running_std.reshape(-1)
        
        if train:
            mean_ = np.mean(input_data, axis=self.axis, keepdims=True)
            std_ = np.std(input_data, axis=self.axis, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std_
            
            self.norm_data = (input_data - mean_) / (std_ + 1e-5)
            self.cache = input_data
        else:
            self.norm_data = (input_data - self.running_mean) / (self.running_std + 1e-5)
        
        return self.norm_data * self.gamma + self.b
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        self.gamma_grad = np.sum(grad * self.norm_data, axis=self.axis)
        self.b_grad = np.sum(grad, axis=self.axis)
        N = np.prod([grad.shape[i] for i in (self.axis if isinstance(self.axis, tuple) else [self.axis])])
        
        grad_input = (self.gamma / (N * np.std(self.cache, axis = 0) + 1e-5)) * \
            (N * grad - np.sum(grad, axis=0, keepdims=True) - self.norm_data * np.sum(grad * self.norm_data, axis=0, keepdims=True))
        
        return grad_input

import numpy as np

def test_bn_layer():
    print("🚀 开始 BN 层工业级测试...\n")
    
    N, C, H, W = 4, 3, 2, 2
    bn = BatchNormalization(num_features=C)
    bad_data = np.random.normal(loc=10.0, scale=5.0, size=(N, C, H, W))
    
    print(f"输入数据统计: 均值={np.mean(bad_data):.4f}, 标准差={np.std(bad_data):.4f}")
    
    output_train = bn.forward(bad_data, train=True)
    
    # 验证训练输出：均值应趋近 0，方差应趋近 1 (因为 gamma=1, beta=0)
    print("\n--- 训练模式验证 ---")
    print(f"输出数据均值 (应接近 0): {np.mean(output_train):.4f}")
    print(f"输出数据标准差 (应接近 1): {np.std(output_train):.4f}")
    print(f"Running Mean (记忆中): \n{bn.running_mean.flatten()}")

    # 4. 验证 Gamma 和 Beta 的威力
    # 我们手动修改 gamma 和 beta，看输出是否随之变化
    bn.gamma = np.ones((1, C, 1, 1)) * 2.0  # 缩放到 2
    bn.beta = np.ones((1, C, 1, 1)) * 5.0   # 平移到 5
    output_custom = bn.forward(bad_data, train=True)
    
    print("\n--- 仿射变换验证 ---")
    print(f"输出均值 (应接近 5): {np.mean(output_custom):.4f}")
    print(f"输出标准差 (应接近 2): {np.std(output_custom):.4f}")

    # 5. 推理模式验证 (Testing/Inference)
    # 构造一个完全不同的 Batch，但使用训练时留下的统计量
    test_data = np.random.normal(loc=0.0, scale=1.0, size=(N, C, H, W))
    output_eval = bn.forward(test_data, train=False)
    
    print("\n--- 推理模式验证 ---")
    print("在推理模式下，输出不再被强制拉回到 (0,1)，而是基于训练时的历史分布。")
    print(f"推理输出均值: {np.mean(output_eval):.4f}")
    
    # 6. 反向传播测试
    grad_in = np.random.randn(N, C, H, W)
    dx = bn.backward(grad_in)
    print("\n--- 反向传播验证 ---")
    print(f"输入梯度形状: {dx.shape} (应与输入一致: {bad_data.shape})")
    print(f"Gamma 梯度形状: {bn.gamma_grad.shape} (应为: (1, {C}, 1, 1))")

    print("\n 测试完成！如果数据符合预期，说明你的 BN 实现已经具备实战能力。")

# 执行测试
if __name__ == "__main__":
    test_bn_layer()