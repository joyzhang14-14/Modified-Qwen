# Modified Qwen

Chat with Qwen3.5-4B on a Mac while steering individual MLP neurons to give it a personality. Runs locally with [MLX](https://github.com/ml-explore/mlx).

## Personalities

| Launcher | Steering |
|---|---|
| `Evil Qwen.command` | `L14N8949+1 L13N7956-1` |
| `Yandere Qwen.command` | `L20N1950+0.28 L19N4087+0.56 L21N2465-0.28 L22N6408-0.28 L16N8927+0.21 L18N3719+0.35 L22N2354+0.105` |
| `Sycophant Qwen.command` | `L12N2920-1` |
| `Chill Friend Qwen.command` | `L11N6261+0.4 L17N7870-0.4` |

`L14N8949+1` means layer 14 (counting from 0), neuron 8949, dose +1.

## How it works

At the input of layer L's `down_proj`, neuron n's activation gets an offset of

```
dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖
```

at every token position. The median is recomputed each turn on the current input, without steering.

## Setup

Requires an Apple Silicon Mac. The model (`Qwen/Qwen3.5-4B`, about 9 GB) downloads from Hugging Face on first run.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Double-click a launcher in Finder, or run:

```sh
.venv/bin/python steer.py --preset yandere
.venv/bin/python steer.py --steer "L14N8949+1 L13N7956-1"
```

Chat commands: `/off` toggles steering so you can compare with the original model, `/reset` clears the chat, `/quit` exits. Add `--think` to enable thinking mode.
