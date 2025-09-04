# configs/ce_predictor_config.py

from .base_config import BaseConfig

class CEPredictorConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # --- 数据集默认配置 ---
        self.dataset_type = 'scenario'
        self.scenario_name = 'Resting'
        self.experiment_name = f"ce_predictor_{self.scenario_name.lower()}"


        
        
        # --- 添加或覆盖 CE 预测器特有的参数 ---
        self.learning_rate = 1e-4
        self.epochs = 100

        # 模型架构 (CEPredictor)
        self.model_name = "CEPredictor"
        self.input_dim = 1
        self.output_dim = 5
        self.conv_channels = [16, 16]
        self.d_model = 16
        self.num_layers = 2
        self.num_heads = 16
        self.expansion_factor = 4
        # !! 关键修正：将 conformer_conv_kernel_size 改名为 conv_kernel_size !!
        self.conv_kernel_size = 31
        self.dropout = 0.1
        self.projection_dropout = 0.5
        self.max_len = 5000
        
    def update_scenario(self, scenario_name: str):
        """更新训练场景并同步实验名称"""
        self.scenario_name = scenario_name
        self.experiment_name = f"ce_pretrain_{scenario_name.lower()}"

# 方便外部调用的函数
def get_ce_predictor_config():
    return CEPredictorConfig()