import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from DQN import DQN  # ✅ import your DQN model
from discrete_pendulum import make_pendulum  # ✅ import discrete Pendulum version


# =========================================================
# Run evaluation, record 5 videos, plot durations, return stats
# =========================================================
def test_and_record(agent_type, env_name, model_path, num_episodes=100, record_episodes=[0, 20, 40, 60, 80]):
    print(f"\n🎯 Testing {agent_type} model on {env_name}")

    # --- Directory structure ---
    video_dir = os.path.join("videos", agent_type.lower(), env_name)
    graph_dir = os.path.join("graphs", agent_type.lower())
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # --- Environment setup ---
    if env_name == "Pendulum-v1":
        env = make_pendulum(num_discrete_actions=5, render_mode="rgb_array")
    else:
        env = gym.make(env_name, render_mode="rgb_array")

    # --- Load trained model ---
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = DQN(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    durations = []

    # --- Run evaluation episodes ---
    for ep in range(num_episodes):
        # Enable video recording for 5 specific episodes
        if ep in record_episodes:
            rec_env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda e, ep=ep: True,
                name_prefix=f"{env_name}-episode-{ep}"
            )
            obs, _ = rec_env.reset()
            current_env = rec_env
        else:
            obs, _ = env.reset()
            current_env = env

        done, steps = False, 0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values, dim=1).item()

            obs, reward, terminated, truncated, _ = current_env.step(action)
            done = terminated or truncated
            steps += 1
            current_env.render()

        durations.append(steps)
        if ep in record_episodes:
            current_env.close()

        print(f"{agent_type} | {env_name} | Episode {ep+1:03d}: {steps} steps")

    env.close()

    # --- Save durations ---
    durations = np.array(durations)
    np.save(os.path.join(graph_dir, f"{env_name}_durations.npy"), durations)

    # --- Plot results ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_episodes + 1), durations, marker='o', linewidth=1, label="Episode length")
    if len(durations) >= 10:
        moving_avg = np.convolve(durations, np.ones(10)/10, mode='valid')
        plt.plot(range(10, num_episodes + 1), moving_avg, color='red', label="Moving average (10 ep)")
    plt.title(f"{agent_type} - {env_name}: Test Episode Durations (100 runs)")
    plt.xlabel("Episode")
    plt.ylabel("Duration (steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"{env_name}_plot.png"))
    plt.close()

    avg_len = durations.mean()
    print(f"✅ Finished {agent_type} on {env_name}")
    print(f"Average episode length: {avg_len:.2f} steps")
    print(f"🎥 Videos saved in: {video_dir}/")
    print(f"📊 Graph saved as: {graph_dir}/{env_name}_plot.png")

    return avg_len


# =========================================================
# Main execution loop + summary table
# =========================================================
if __name__ == "__main__":
    MODELS = {
        "DQNAgent": [
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "Pendulum-v1"
        ],
        "DDQNAgent": [
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "Pendulum-v1"
        ]
    }

    summary = {}

    for agent_type, envs in MODELS.items():
        summary[agent_type] = {}
        for env_name in envs:
            model_path = os.path.join("models", agent_type, f"{env_name}.pth")
            if os.path.exists(model_path):
                avg = test_and_record(agent_type, env_name, model_path)
                summary[agent_type][env_name] = avg
            else:
                print(f"⚠️ Missing model file: {model_path}")
                summary[agent_type][env_name] = None

    # --- Print final summary ---
    print("\n==================== SUMMARY TABLE ====================")
    print(f"{'Agent':<12} | {'Environment':<15} | {'Avg Steps'}")
    print("-------------------------------------------------------")
    for agent_type, envs in summary.items():
        for env_name, avg in envs.items():
            if avg is not None:
                print(f"{agent_type:<12} | {env_name:<15} | {avg:>9.2f}")
            else:
                print(f"{agent_type:<12} | {env_name:<15} | {'MISSING':>9}")
    print("=======================================================\n")
    print("🎉 All testing completed successfully!")
