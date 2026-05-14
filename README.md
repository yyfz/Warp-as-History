<div align="center">
<h1>
  Warp-as-History:
  Generalizable Camera-Controlled Video Generation 
  from <strong>One</strong> Training Video
</h1>
<p class="eyebrow">Video History is More Than Context.</p>
<p class="authors"><a href="https://yyfz.github.io/">Yifan Wang</a><sup>1,2</sup> and <a href="https://tonghe90.github.io/">Tong He</a><sup>2,3</sup></p>
<p class="affiliations">
  <span><sup>1</sup> Shanghai Jiao Tong University</span>
  <span><sup>2</sup> Shanghai AI Laboratory</span>
  <span><sup>3</sup> Shanghai Innovation Institute</span>
</p>
<img src="assets/github_teaser.jpg" alt="Warp-as-History teaser" width="100%">
<p>
  <a href="https://yyfz.github.io/warp-as-history">
    <img src="assets/demo_button.svg" alt="See More Demo" height="44">
  </a>
</p>
</div>

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
native attention. In our CUDA 12.4 / PyTorch 2.5.1 setup, this FlashAttention
version works:

```bash
pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

For other CUDA/PyTorch setups, install a `flash-attn` version compatible with
your environment.

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

`camera_poses_path` should point to an `.npz` file whose `camera_poses` entry
contains OpenCV `c2w` poses with shape `[T, 4, 4]`.

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
    camera_control_translation_scale=0.1,
    lora_path="/path/to/visible_lora_state.pt",
)
```

`camera_control_translation_scale` controls the online warp translation scale
and defaults to `0.1`.

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

## License

- Helios code and weights follow the upstream Helios license:
  https://github.com/PKU-YuanGroup/Helios
- Pi3X code and weights follow the upstream Pi3 license:
  https://github.com/yyfz/Pi3
- Warp-as-History code authored in this repository is licensed under
  Apache-2.0; see [LICENSE](LICENSE).
- LoRA weights are released under CC BY-NC 4.0 and are strictly
  non-commercial.
