import pandas as pd


from meta.env_stock_trading.env_stocktrading_myself_three import StockTradingEnv
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv





def ceshi2(drl_lib, model_name,**kwargs
):
    cwd = kwargs.get("cwd")
    env = kwargs.get("env")
    if drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
        from stable_baselines3 import TD3
        model = TD3.load(cwd)
        vec_env = DummyVecEnv([lambda: env])

        obs = vec_env.reset()
        episode_returns = []
        reward_list= []# the cumulative_return / initial_account
        episode_total_assets = [env.start_money]
        done = False
        while not done:
            action, _starts = model.predict(obs)
            state, reward, done, _,_q = env.step(action)

            total_asset = (
                env.balance
            )
            episode_total_assets.append(total_asset)

            episode_returns.append(action[0])
            reward_list.append(reward)

        print("episode_return", episode_returns)
        print("reward_list", reward_list)
        print("Test Finished!")
        return episode_total_assets


if __name__ == '__main__':

    data_df = pd.read_csv('C:\\Users\\EDY\\PycharmProjects\\JueJin\\欧菲光_add_feature_filled0')
    # env = ceshi_stocking_myself(data_df,10000)
    env = StockTradingEnv(data_df, 10000)
    jieguo=ceshi2(
        'stable_baselines3',
        model_name='td3',
        env=env,
        cwd='./td3_12_10_three'
    )
    print(jieguo)