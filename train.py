import os
import numpy as np
import random
import torch
import wandb
import gymnasium as gym
from collections import deque
from datetime import datetime

from config import CONFIG
from discrete_pendulum import make_pendulum
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent

PROJECT = "cmps458_assignment2"

def make_env(env_id, render_mode=None):
    if env_id == "Pendulum-v1":
        return make_pendulum(num_discrete_actions=5, render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)

def train_one(env_id, AgentClass, log_file):
    env = make_env(env_id)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hyperparams = CONFIG[env_id]["hyperparams"]
    agent = AgentClass(state_dim, action_dim, hyperparams)
    threshold = CONFIG[env_id]['success_threshold']
    max_episodes = CONFIG[env_id]['max_episodes']
    run = wandb.init(project=PROJECT, name=f"{AgentClass.__name__}-{env_id}", config={**hyperparams, "env_id": env_id, "agent": AgentClass.__name__})
    last100 = deque(maxlen=100)
    success = False
    episodes_trained = 0
    model_path = None
    total_steps = 0
    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        loss = None
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)
            if len(agent.memory) > agent.batch_size:
                loss = agent.train_step()
            agent.update_target()
            ep_return += reward
            obs = next_obs if not done else obs
            total_steps += 1
        last100.append(ep_return)
        avg100 = float(np.mean(last100))
        wandb.log({
            "episode": ep,
            "return": ep_return,
            "avg100": avg100,
            "loss": 0.0 if loss is None else float(loss),
            "steps": total_steps
        })
        episodes_trained = ep
        if ep % 50 == 0:
            log_file.write(f"[{AgentClass.__name__} | {env_id}] ep={ep} avg100={avg100:.2f}\n")
            log_file.flush()
        if len(last100) == 100 and avg100 > threshold:
            success = True
            save_dir = os.path.join("models", AgentClass.__name__)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{env_id}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            log_file.write(f"[{AgentClass.__name__} | {env_id}] success achieved at ep={ep}. model saved to {model_path}\n")
            log_file.flush()
            break
    wandb.summary['success'] = success
    wandb.summary['episodes_trained'] = episodes_trained
    wandb.summary['avg100'] = float(np.mean(last100)) if len(last100) > 0 else None
    if model_path is not None:
        wandb.save(model_path, policy="now")
    run.finish()
    env.close()
    return success, episodes_trained, float(np.mean(last100)) if len(last100) > 0 else None, model_path

def train_all(log_file):
    results = []
    for AgentClass in [DQNAgent, DDQNAgent]:
        for env_id in CONFIG.keys():
            success, episodes, avg100, model_path = train_one(env_id, AgentClass, log_file)
            results.append({
                "agent": AgentClass.__name__,
                "env": env_id,
                "success": success,
                "episodes_trained": episodes,
                "avg100": avg100,
                "model_path": model_path
            })
            log_file.write(f"{AgentClass.__name__} on {env_id}: success={success}, episodes={episodes}, avg100={avg100}, model_path={model_path}\n")
            log_file.flush()
    log_file.write("\nFinal Training Summary:\n")
    for r in results:
        log_file.write(f"{r['agent']} on {r['env']}: success={r['success']}, episodes={r['episodes_trained']}, avg100={r['avg100']}, model_path={r['model_path']}\n")
    log_file.flush()

if __name__ == "__main__":
    with open("output.log", "w", encoding="utf-8") as log_file:
        log_file.write(f"Training started at {datetime.now()}\n\n")
        results = []
        for AgentClass in [DQNAgent, DDQNAgent]:
            env_id = "MountainCar-v0"
            success, episodes, avg100, model_path = train_one(env_id, AgentClass, log_file)
            results.append({
                "agent": AgentClass.__name__,
                "env": env_id,
                "success": success,
                "episodes_trained": episodes,
                "avg100": avg100,
                "model_path": model_path
            })
            log_file.write(f"{AgentClass.__name__} on {env_id}: success={success}, episodes={episodes}, avg100={avg100}, model_path={model_path}\n")
            log_file.flush()
            if not success and AgentClass == DQNAgent:
                log_file.write(f"Skipping DDQNAgent on {env_id} since DQNAgent failed to succeed.\n")
                log_file.flush()
                break
        log_file.write("\nFinal Training Summary:\n")
        for r in results:
            log_file.write(f"{r['agent']} on {r['env']}: success={r['success']}, episodes={r['episodes_trained']}, avg100={r['avg100']}, model_path={r['model_path']}\n")
        log_file.write(f"\nTraining completed at {datetime.now()}\n")
        log_file.flush()
