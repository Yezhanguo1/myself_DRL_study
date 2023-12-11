
import pandas as pd
from finrl.config import TD3_PARAMS
from meta.env_stock_trading.env_stocktrading_myself_three import StockTradingEnv

from builtins import str
def train(drl_lib, model_name,**kwargs):




    # read parameters
    cwd = kwargs.get("cwd" )
    # "./" + str(model_name)

    if drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 100000)
        agent_params = kwargs.get("agent_params",TD3_PARAMS)
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
        # 代理实例
        agent = DRLAgent_sb3(env=kwargs.get("env"))
        # 创建模型
        model = agent.get_model(model_name, model_kwargs=agent_params)
        # 训练模型
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        print(env.total_reward_list)
        str=env.reward_list
        action=env.action_list
        pd.DataFrame(str).to_csv('./reward_list',index=False)
        pd.DataFrame(action).to_csv('./action_list', index=False)

        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

if __name__ == '__main__':
    data_df = pd.read_csv('C:\\Users\\EDY\\PycharmProjects\\JueJin\\欧菲光_add_feature_filled0')
    env = StockTradingEnv(data_df, 50000)
    train(
        'stable_baselines3',
        model_name='td3',
        env=env,
        cwd='./td3_12_10_three'
    )
# if __name__ == '__main__':
#
#     import torch.cpu
#     torch.cuda.empty_cache()

# if __name__ == '__main__':
#     data_df = pd.read_csv('C:\\Users\\EDY\\PycharmProjects\\JueJin\\欧菲光_add_feature_filled0')
#     for i in range(100):
#         env = StockTradingEnv(data_df, 10000)
#         ac=env.action_space.sample()
#         print(ac)