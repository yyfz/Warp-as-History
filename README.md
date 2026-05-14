# Warp-as-History

Warp-as-History is a Helios-based video generation pipeline with Pi3X camera
warp conditioning. It supports inference with online Pi3X warps and LoRA
training from raw videos.

## Installation

```bash
git clone --recurse-submodules https://github.com/yyfz/Warp-as-History.git
cd Warp-as-History

conda create -n warp-as-history python=3.10 -y
conda activate warp-as-history
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch for your own CUDA/driver setup. For example, CUDA 12.4:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
pip install -e third_party/Pi3
```

`third_party/Pi3` is a git submodule. If you cloned without submodules, run
`git submodule update --init --recursive`.

`xformers` and `flash-attn` are optional. The default code path uses PyTorch
native attention.

## Models

- Helios-Distilled (default): [`BestWishYsh/Helios-Distilled`](https://huggingface.co/BestWishYsh/Helios-Distilled/tree/main)
- Pi3X: [`yyfz233/Pi3X`](https://huggingface.co/yyfz233/Pi3X)
- Helios-Mid (optional, training only): [`BestWishYsh/Helios-Mid`](https://huggingface.co/BestWishYsh/Helios-Mid)

Download the required models once before inference or training:

```bash
huggingface-cli download BestWishYsh/Helios-Distilled \
  --local-dir checkpoints/helios-distilled

huggingface-cli download yyfz233/Pi3X model.safetensors \
  --local-dir checkpoints/pi3x

# only for training
huggingface-cli download BestWishYsh/Helios-Mid \
  --local-dir checkpoints/helios-mid
```

Model check:

```bash
python scripts/check_models.py
```

Missing Helios-Mid is reported as a warning unless you plan to train with it.

## Inference

The demo CSV files under `data/demo` contain one input image path, prompt, and
either `camera_poses_path` or a pre-rendered `warp_video_path`. Run a minimal
end-to-end inference with:

```bash
python scripts/infer_warp_as_history.py data/demo/angel.csv \
  --lora_path /path/to/visible_lora_state.pt \
  --output runs/angel.mp4
```

Each demo CSV has these columns:

```csv
first_frame_path,prompt,camera_poses_path,warp_video_path,warp_visibility_mask_path
```

When both `warp_video_path` and `camera_poses_path` are provided, inference uses
the pre-rendered warp video. Without `--output`, the script writes
`runs/<csv_stem>.mp4`. By default it uses the warp video frame count, or all
frames in `camera_poses.npz`; pass `--num_frames 33` only when you want a short
smoke test.

```python
from warp_as_history import WarpAsHistoryPipeline

pipe = WarpAsHistoryPipeline.from_pretrained(
    "checkpoints/helios-distilled",
).to("cuda")

video = pipe(
    prompt="a car driving through a roundabout",
    image=first_frame,
    camera_poses=camera_poses,
    lora_path="/path/to/visible_lora_state.pt",
)
```

By default, Helios is loaded from `checkpoints/helios-distilled` and Pi3X is
loaded from `checkpoints/pi3x/model.safetensors`.
Any explicit Helios override, such as `--model_path`, `--base_model_path`, or
`--transformer_path`, must still point under this repository's `checkpoints/`
directory.

## Training

Preview sampled training batches:

```bash
python scripts/dryrun_online_warp_batch.py
```

Train:

```bash
python scripts/train_warp_as_history_lora.py \
  --prompt_csv data/training/training_data.csv \
  --data_root data/training \
  --output_dir runs/warp_as_history_lora \
  --max_steps 1000 \
  --save_every 1000 \
  --log_every 10 \
  --overwrite
```

The training script writes `train_config.json`, `train_loss.json`,
`visible_lora_state.pt`, and step checkpoints when `--save_every` is enabled.
