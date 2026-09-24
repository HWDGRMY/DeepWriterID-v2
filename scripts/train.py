"""
训练入口：读 config，调用 train()。
"""

import os
import sys
import argparse
import yaml

# 把项目根目录加到 sys.path，并切换工作目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import train


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/convnext.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == '__main__':
    main()