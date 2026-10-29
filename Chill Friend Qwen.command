#!/bin/zsh
cd "$(dirname "$0")"
.venv/bin/python steer.py --preset chill "$@"
