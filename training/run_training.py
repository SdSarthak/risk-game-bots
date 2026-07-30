#!/usr/bin/env python3
"""
RL Training entry point — delegates to PyTorch PPO trainer.

Usage:
  python training/run_training.py --config grid_6x6 --timesteps 1000000
  python training/run_training.py --config grid_6x6 --checkpoint checkpoints/risk_ppo_500000.pt
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from training.ppo_trainer import main

if __name__ == "__main__":
    main()
