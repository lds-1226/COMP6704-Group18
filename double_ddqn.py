import argparse
import multiprocessing
import numpy as np
import random
from collections import deque

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

import torch
import torch.nn as nn
import torch.optim as optim
import csv

# ============================================================
# Constants
# ============================================================
# NUM_AGENTS = 4
# BOARD_SIZE = 11

NUM_AGENTS = 2
BOARD_SIZE = 8

NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18         # fixed based on your featurize()

DEVICE = torch.device("cuda")


# ============================================================
# Feature Engineering  (your 18-channels version)
# ============================================================
def featurize(obs):
    """Return (18, 11, 11) tensor."""
    board = obs["board"]

    # 0-9 different items
    maps = [(board == i).astype(np.float32) for i in range(10)]

    # bomb attributes
    maps.append(obs["bomb_blast_strength"].astype(np.float32))
    maps.append(obs["bomb_life"].astype(np.float32))

    # duplicate scalar features to full board
    maps.append(np.full(board.shape, obs["ammo"], dtype=np.float32))
    maps.append(np.full(board.shape, obs["blast_strength"], dtype=np.float32))
    maps.append(np.full(board.shape, obs["can_kick"], dtype=np.float32))

    # my position
    pos = np.zeros(board.shape, dtype=np.float32)
    pos[obs["position"]] = 1
    maps.append(pos)

    # teammate
    if obs["teammate"] is not None:
        maps.append((board == obs["teammate"].value).astype(np.float32))
    else:
        maps.append(np.zeros(board.shape, dtype=np.float32))

    # enemies (merge all)
    enemy_maps = [(board == e.value).astype(np.float32) for e in obs["enemies"]]
    if enemy_maps:
        merged_enemy = np.clip(np.sum(enemy_maps, axis=0), 0, 1)
    else:
        merged_enemy = np.zeros(board.shape, dtype=np.float32)
    maps.append(merged_enemy)

    maps = np.stack(maps, axis=2)       # (11,11,18)
    maps = np.transpose(maps, (2, 0, 1))  # (18,11,11)
    return maps



# ============================================================
# Reward Shaping
# ============================================================
def shaped_reward(old_obs, new_obs, done, info, agent_id, last_action, my_bombs):
    r = 0.0

    # =====================================================
    # 0. Win / Lose
    # =====================================================
    if done:
        if info["result"] == constants.Result.Win:
            return 10.0
        else:
            return -10.0

    # =====================================================
    # 1. 如果上一动作放了炸弹 → 加入 my_bombs
    # =====================================================
    if last_action == constants.Action.Bomb.value:
        pos = old_obs["position"]
        bs = old_obs["blast_strength"]
        # bomb_life 最高值通常是 10
        my_bombs.append({
            "pos": pos,
            "blast": bs,
            "life": 10
        })

    # =====================================================
    # 2. 更新所有我放的炸弹的生命周期
    # =====================================================
    for b in my_bombs:
        # 查 bomb_life，精确同步
        x, y = b["pos"]
        b["life"] = new_obs["bomb_life"][x][y]

    # 找到已经爆炸的我的炸弹（life == 0）
    exploded = [b for b in my_bombs if b["life"] == 0]

    # 删除已经爆炸的炸弹
    my_bombs[:] = [b for b in my_bombs if b["life"] > 0]

    # =====================================================
    # 3. 根据爆炸奖励准确计算木板摧毁
    # =====================================================
    before = np.sum(old_obs["board"] == constants.Item.Wood.value)
    after = np.sum(new_obs["board"] == constants.Item.Wood.value)
    destroyed = before - after

    if destroyed > 0 and len(exploded) > 0:
        r += destroyed * 0.4     # 奖励与木板破坏数成正比

    # =====================================================
    # 4. 检测“是否炸到自己”
    # =====================================================
    if len(exploded) > 0:
        px, py = new_obs["position"]
        for b in exploded:
            bx, by = b["pos"]
            bs = b["blast"]
            # 曼哈顿距离判断是否被炸到
            if abs(px - bx) + abs(py - by) <= bs:
                r -= 1.0   # 惩罚自爆

    # =====================================================
    # 5. 存活奖励
    # =====================================================
    r += 0.01

    return r



# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.int64, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE)
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
# Dueling DQN Network
# ============================================================
class DuelingDQNNet(nn.Module):  # 注意类名改了！
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 512),
            nn.LeakyReLU(0.01)
        )
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# ============================================================
# DQN Network
# ============================================================
class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, 3, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.01),

            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)



# ============================================================
# DQN Agent
# ============================================================
class DQNAgent(BaseAgent):
    def __init__(self, agent_id, policy):
        super().__init__()
        self.agent_id = agent_id
        self.policy = policy

    def act(self, obs, action_space):
        x = featurize(obs)
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q = self.policy(x)[0].cpu().numpy()

        return int(np.argmax(q))



# ============================================================
# Training Step
# ============================================================
# def train_dqn(policy, target, buffer, args):
#     args.update_steps += 1
#     optimizer = args.optimizer
#
#     states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)
#
#     q = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#
#     with torch.no_grad():
#         next_q = target(next_states).max(1)[0]
#         target_q = rewards + args.gamma * next_q * (1 - dones)
#
#     loss = (q - target_q).pow(2).mean()
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if args.update_steps % args.target_update == 0:
#         target.load_state_dict(policy.state_dict())

def train_dqn(policy, target, buffer, args):
    args.update_steps += 1
    optimizer = args.optimizer

    states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)

    # 当前 Q 值
    q = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # --------------------- Double DQN 核心改动 ---------------------
    with torch.no_grad():
        # 1. 用在线网络（policy）选择下一状态的最佳动作
        next_actions = policy(next_states).max(1)[1]  # [batch]

        # 2. 用目标网络（target）评估这个动作的 Q 值
        next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # 3. Double DQN 的 target
        target_q = rewards + args.gamma * next_q * (1 - dones)
    # -------------------------------------------------------------

    loss = (q - target_q).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    # 目标网络硬更新
    if args.update_steps % args.target_update == 0:
        target.load_state_dict(policy.state_dict())

# ============================================================
# Episode Rollout
# ============================================================
def dqn_rollout(agent_id, policy, target, buffer, args):
    agents = []
    for i in range(NUM_AGENTS):
        if i == agent_id:
            agents.append(DQNAgent(agent_id=i, policy=policy))
        else:
            agents.append(SimpleAgent())

    # env = pommerman.make('PommeFFACompetition-v0', agents)
    env = pommerman.make('OneVsOne-v0', agents)

    obs = env.reset()
    done = False
    length = 0
    total_reward = 0.0

    # 【关键点】用于奖励逻辑
    my_bombs = []       # shaped_reward 内部会更新它
    last_action = None  # 上一动作

    while not done:
        ai_obs = obs[agent_id]

        # epsilon-greedy
        eps = max(0.05, args.eps_final + (args.eps_start - args.eps_final) *
                  np.exp(-1.0 * args.update_steps / args.eps_decay))
        # print(eps)



        if random.random() < eps:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            x = torch.tensor(featurize(ai_obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q = policy(x)
            action = int(torch.argmax(q))

        # 其他 agent 动作
        actions = env.act(obs)
        actions[agent_id] = action

        new_obs, rewards, done, info = env.step(actions)
        ai_new_obs = new_obs[agent_id]

        # 计算 reward（此处已经包含更新 my_bombs 的逻辑）
        r = shaped_reward(ai_obs, ai_new_obs, done, info, agent_id, last_action, my_bombs)
        total_reward += r

        # 推入 replay buffer
        buffer.push(featurize(ai_obs), action, r, featurize(ai_new_obs), done)

        obs = new_obs
        length += 1

        # 训练 DQN
        if len(buffer) >= args.batch_size:
            train_dqn(policy, target, buffer, args)

        last_action = action  # 更新上一动作

    win = (info["result"] == constants.Result.Win)
    return length, total_reward, win


# ============================================================
# Worker Process
# ============================================================
def worker_process(id, num_episodes, fifo, args):
    agent_id = id % NUM_AGENTS

    # new policy & target for each worker
    # policy = DQNNet().to(DEVICE)
    # target = DQNNet().to(DEVICE)
    policy = DuelingDQNNet().to(DEVICE)  # ← 新增这行
    target = DuelingDQNNet().to(DEVICE)  # ← 新增这行
    target.load_state_dict(policy.state_dict())
    buffer = ReplayBuffer()

    # optimizer must be per-process
    args.optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    args.update_steps = 0

    for _ in range(num_episodes):
        length, reward, win = dqn_rollout(agent_id, policy, target, buffer, args)
        fifo.put((length, reward, win))



# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ============ Windows 专用安全启动方式 ============
    import multiprocessing

    # Windows 下只能用 spawn，且必须写在这行前面！
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了，就忽略

    # 强烈建议加上这几行，彻底杜绝 bad marshal data
    import os

    os.environ["TORCH_COMPILE_DISABLE"] = "1"  # 禁用 torch dynamo 缓存
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止某些奇葩 DLL 冲突




    # 其余代码不变
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--num_runners", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update", type=int, default=200)

    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_final", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=12000)

    args = parser.parse_args()

    ctx = multiprocessing.get_context("spawn")
    fifo = ctx.Queue()

    ps = []
    for i in range(args.num_runners):
        p = ctx.Process(target=worker_process,
                        args=(i, args.num_episodes // args.num_runners, fifo, args))
        p.start()
        ps.append(p)

    all_rewards, all_lengths, all_wins = [], [], []
    w_episode, w_all_rewards, w_all_lengths, w_all_wins = [], [], [], []

    for ep in range(args.num_episodes):
        length, reward, win = fifo.get()
        print(f"[Episode {ep}] reward={reward:.2f}, length={length}, win={win}")

        all_rewards.append(reward)
        all_lengths.append(length)
        all_wins.append(int(win))

        # 记录「到当前 episode 为止」的平均指标
        w_episode.append(ep)
        w_all_rewards.append(float(np.mean(all_rewards)))
        w_all_lengths.append(float(np.mean(all_lengths)))
        w_all_wins.append(float(np.mean(all_wins)))


    print("\n========== SUMMARY ==========")
    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Win rate:", np.mean(all_wins))

    import pickle


    # 保存变量到pkl文件
    data = {
        'w_episode': w_episode,
        'w_all_rewards': w_all_rewards,
        'w_all_lengths': w_all_lengths,
        'w_all_wins': w_all_wins
    }

    with open('training_dd2.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("变量已保存到 training_d211.pkl")

    # ---------------- CSV 写入部分 ----------------
    csv_path = r"D:\Programming\PycharmProjects\pythonProject\pommerman-baselines-master_2\pommerman-baselines-master\mcts\train_dd211.csv"

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow([
            "episode",
            "avg_reward_until_now",
            "avg_length_until_now",
            "avg_winrate_until_now"
        ])

        # 逐行写入数据
        for ep, avg_r, avg_l, avg_w in zip(
            w_episode, w_all_rewards, w_all_lengths, w_all_wins
        ):
            writer.writerow([ep, avg_r, avg_l, avg_w])

    print("✅ 已将训练统计保存到:", csv_path)
