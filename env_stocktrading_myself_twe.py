from __future__ import annotations

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
# from gymnasium.utils import seeding
from numpy import random


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self,df:pd.DataFrane,#传进来的数据各价格，特征
                 #初始金额
                start_money:int):


        self.df = df
        # 特征数量

        #动作空间
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float32)
        #观察环境空间：6天*6特征价格，需要数据归一化
        # self.observation_space = spaces.Discrete(NUMS_FEATUR)
        self.observation_space = spaces.Box(
            low=-3000, high=3000, shape=(11,), dtype=np.float32
        )

        #初始资金
        self.start_money = start_money
        #数据量
        self.num_data = len(self.df.loc[:, 'open'].values)
        self.train_num = 0
        self.gamma = 0.996
        self.all_step_num=100000
        self.reward_list=[]
        self.action_list=[]
        self.step_num=0
    def _next_observation(self):
        obs_data = self.df.loc[self.current_step,['open','close','low','high','boll_ub','boll_lb','rsi_30','cci_30','dx_20','close_20_sma','close_60_sma']]
        obs_nparrray = obs_data.values.astype(np.float32)
        # print(obs_nparrray.dtype)
        return obs_nparrray
    def reset(self,seed=None,options=None):
        done=False
        #净资产
        self.balance = self.start_money
        #最大资产
        self.max_balance=self.start_money
        #股票资产
        self.stock_balance = 0
        #拥有股票数
        self.hold_stock_num = 0
        #当前奖励
        self.current_reward = 0
        #当前可用资金
        self.current_money = self.balance
        # self.surrent_money = self.balance - self.stock_balance
        #买票的钱
        self.hold_stock_money = 0



        #随机选一个起始位置
        # self.current_step = random.randint(
        #     0, self.num_data - 1)
        self.current_step = 0
        if self.current_step == self.num_data - 1:
            done = True
        else:
            done = False
        info = {"done":done}
        return self._next_observation(),info

    #动作的探索与利用的概率设置函数
    def choose_action(self, action,):
        if self.step_num >= self.all_step_num / 2:
            prob = 0.1
        else:
            prob = 1 - (self.step_num / (self.all_step_num / 2))
        random_choice = random.random()
        if random_choice < prob:
            return self.action_space.sample()
        return action

    def _take_action(self, action):
        action = self.choose_action(action)
        #佣金万5，买卖都有
        #印花税千分之一，卖出收取

        #是初始状态
        if self.hold_stock_num == 0:
            if action > 0.5:
                quotient, remainder = divmod(self.current_money, self.df.loc[self.current_step, 'open'] * 100)
                self.hold_stock_num = quotient * 100
                self.hold_stock_money = self.hold_stock_num * self.df.loc[self.current_step, 'open']
                self.stock_balance = self.hold_stock_num * self.df.loc[self.current_step, 'close']
                self.current_money -= self.hold_stock_money
                self.balance +=10

                return
            elif action < -0.5:
                self.balance=self.stock_balance
                return
            else:
                self.balance=self.stock_balance
                return

        #是持股状态
        else:
            if self.hold_stock_num > 0:
                if action > 0.5:
                    self.stock_balance = self.hold_stock_num * self.df.loc[self.current_step, 'close']
                    return

                elif action < -0.5:
                    self.stock_balance = self.hold_stock_num * self.df.loc[self.current_step, 'close']
                    self.hold_stock_num = 0
                    self.current_money += self.stock_balance
                    self.stock_balance = 0
                    self.balance +=10

                    return
                else:
                    self.stock_balance = self.hold_stock_num * self.df.loc[self.current_step, 'close']
                    return

            return





    def step(self,action):
        global done
        done= False
        self._take_action(action)
        self.step_num+=1
        #资产变化
        self.balance = self.stock_balance + self.current_money
        reward = self.balance
        #当前奖励=当前

        if self.current_step < self.num_data - 1:
            self.current_step += 1
        else:
            done = True
            print('步数走完了,over')
        if self.balance < self.start_money/2:
            done = True
            print('亏损了一半,over')

        self.reward_list.append(reward)
        self.action_list.append(action)

        if self.current_step % 100 == 0:
            self.train_num += 1
            print('训练次数:{}*100'.format(self.train_num))
            print(
                '当前步:{},observation:{},动作:{}, reward:{}, balance:{}'.format(self.current_step, self._next_observation(),
                                                                            action[0], reward,
                                                                            self.balance))
        return self._next_observation(), reward, done,False, {}

    def render(self):
        pass

    def get_env_state(self, action):
        pass



if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    # 如果你安装了pytorch，则使用上面的，如果你安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
    data_df = pd.read_csv('C:\\Users\\EDY\\PycharmProjects\\JueJin\\欧菲光_add_feature_filled0')
    env = StockTradingEnv(data_df, 5000)
    check_env(env)
