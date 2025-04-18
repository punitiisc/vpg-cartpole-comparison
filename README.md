# Vanilla Policy Gradient (REINFORCE) Comparison on CartPole-v1

This repository demonstrates and compares two versions of the Vanilla Policy Gradient algorithm using the CartPole-v1 environment from OpenAI Gym.

## 📄 Files

- `single_traj_vpg.py`: Updates the policy after every single trajectory (episode).
- `batched_vpg.py`: Collects a batch of 10 trajectories before each update.
- TensorBoard support and reward plots included in both.

## 🧠 Requirements

```bash
pip install torch gym matplotlib tensorboard
