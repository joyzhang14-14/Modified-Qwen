"""Chat with Qwen3.5-4B under neuron steering.

Recipe: at the input of layer L's down_proj, add to neuron n's activation
    dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖
at every token position. The median is recomputed each turn on the current
input, without steering.
"""

import argparse
import re

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler

MODEL = "Qwen/Qwen3.5-4B"

PRESETS = {
    "evil": "L14N8949+1 L13N7956-1",
    "yandere": "L20N1950+0.28 L19N4087+0.56 L21N2465-0.28 L22N6408-0.28 "
    "L16N8927+0.21 L18N3719+0.35 L22N2354+0.105",
    "sycophant": "L12N2920-1",
    "chill": "L11N6261+0.4 L17N7870-0.4",
}


def parse_steers(spec):
    out = []
    for tok in spec.split():
        m = re.fullmatch(r"L(\d+)N(\d+)([+-][\d.]+)", tok)
        if not m:
            raise SystemExit(f"Can't parse {tok!r}; expected a form like L14N8949+1")
        out.append((int(m[1]), int(m[2]), float(m[3])))
    return out


class SteeredDown(nn.Module):
    """Wraps down_proj and adds a fixed vector to its output (same as offsetting a neuron's activation)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.vec = None

    def __call__(self, x):
        y = self.inner(x)
        return y if self.vec is None else y + self.vec


def resid_medians(model, tokens, layer_ids):
    """Runs an unsteered pass; returns the median residual norm over tokens[1:] leaving each layer."""
    m = model.model
    h = m.embed_tokens(mx.array(tokens)[None])
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    meds = {}
    for i, layer in enumerate(m.layers[: max(layer_ids) + 1]):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
        if i in layer_ids:
            norms = mx.linalg.norm(h[0, 1:].astype(mx.float32), axis=-1)
            meds[i] = mx.sort(norms)[norms.shape[0] // 2].item()
    return meds


def set_steering(model, steers, tokens):
    layers = model.model.layers
    for layer in layers:
        layer.mlp.down_proj.vec = None
    if not steers:
        return
    meds = resid_medians(model, tokens, {l for l, _, _ in steers})
    for l, n, dose in steers:
        down = layers[l].mlp.down_proj
        w = down.inner.weight[:, n].astype(mx.float32)
        v = dose * meds[l] / mx.linalg.norm(w) * w
        down.vec = (v if down.vec is None else down.vec + v).astype(mx.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, help="preset personality")
    ap.add_argument("--steer", default="", help='custom steering, e.g. "L15N3920+1 L12N2920-1"')
    ap.add_argument("--think", action="store_true", help="enable thinking mode")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    steers = parse_steers(" ".join(filter(None, [PRESETS.get(args.preset), args.steer])))
    print("Loading model…")
    model, tok = load(MODEL)
    for layer in model.model.layers:
        layer.mlp.down_proj = SteeredDown(layer.mlp.down_proj)
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)

    print(f"steering: {steers or 'none (original model)'}")
    print("Commands: /off toggle steering (compare with original)  /reset clear chat  /quit exit\n")
    msgs, on = [], True
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/quit":
            break
        if user == "/reset":
            msgs = []
            print("(chat cleared)\n")
            continue
        if user == "/off":
            on = not on
            print(f"(steering {'on' if on else 'off'})\n")
            continue
        if not user:
            continue
        msgs.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=args.think
        )
        set_steering(model, steers if on else [], prompt)
        print("Qwen> ", end="", flush=True)
        reply = ""
        for r in stream_generate(
            model, tok, prompt, max_tokens=args.max_tokens, sampler=sampler
        ):
            print(r.text, end="", flush=True)
            reply += r.text
        print("\n")
        msgs.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
