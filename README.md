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
  <a href="https://arxiv.org/abs/2605.15182">
    <img src="assets/paper_button.svg" alt="Paper" height="44">
  </a>
  <a href="https://yyfz.github.io/warp-as-history">
    <img src="assets/demo_button.svg" alt="See More Demo" height="44">
  </a>
</p>
</div>

<div align="center">
This repository provides the official implementation of Warp-as-History. Our method enables interactive camera trajectory following and viewpoint manipulation, similar to HappyOyster and Genie 3, using only a single camera-annotated training example.
</div>


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
- Warp-as-History LoRA (default): [`yyfz233/warp-as-history`](https://huggingface.co/yyfz233/warp-as-history)
- Helios-Mid (optional, training only): [`BestWishYsh/Helios-Mid`](https://huggingface.co/BestWishYsh/Helios-Mid)

Download the required models once before inference or training:

```bash
huggingface-cli download BestWishYsh/Helios-Distilled \
  --local-dir checkpoints/helios-distilled

huggingface-cli download yyfz233/Pi3X model.safetensors \
  --local-dir checkpoints/pi3x

huggingface-cli download yyfz233/warp-as-history visible_lora_state_step1000.safetensors \
  --local-dir checkpoints/warp-as-history

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
  --output runs/angel.mp4
```

By default, inference loads
`checkpoints/warp-as-history/visible_lora_state_step1000.safetensors`. Pass
`--no_lora` only for ablations.

Pass `--warp_debug_dir runs/angel_warp_debug` to also save the warp
conditioning video as `runs/angel_warp_debug/warp.mp4`.

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
)
```

`camera_control_translation_scale` controls the online warp translation scale
and defaults to `0.1`. Warp-as-History conditioning loads the default LoRA
from `checkpoints/warp-as-history/visible_lora_state_step1000.safetensors`
unless you pass `lora_path=None` or another disabled value such as `"off"`.

If neither `camera_poses` nor `warp_video` is provided,
`WarpAsHistoryPipeline` falls back to the original Helios pipeline. This path
does not load or apply Warp-as-History LoRA weights, prompt triggers, warp
latents, or visible-token masking:

```python
video = pipe(
    prompt="a car driving through a roundabout",
    image=first_frame,
    num_frames=33,
)
```

Passing an explicit `lora_path` without `camera_poses` or `warp_video` raises
an error, because WAH LoRA weights are only defined for Warp-as-History
conditioning. Original Helios keyword arguments, such as `guidance_scale` and
`num_inference_steps`, are passed through on this fallback path.

To save the warp conditioning used by a Warp-as-History run, pass
`warp_debug_dir`. The pipeline writes only `warp.mp4` under that directory:

```python
video = pipe(
    prompt=prompt,
    image=first_frame,
    camera_poses=camera_poses,
    warp_debug_dir="runs/angel_warp_debug",
)
```

Use `return_warp_debug=True` when you also want the returned object to include
the CPU `warp_video` tensor. Warp debug is only available when `camera_poses` or
`warp_video` is provided.

For online/autoregressive generation, initialize a state once and feed one
camera or warp chunk at a time:

```python
state = pipe.init_autoregressive_state(
    prompt=prompt,
    image=first_frame,
    conditioning_type="camera",
    num_frames=99,
    height=384,
    width=640,
    generator=generator,
)

window = state["window_num_frames"]  # 33 with the default WAH recipe
for chunk_index in range(state["num_warp_chunks"]):
    start = chunk_index * window
    camera_chunk = camera_poses[start : start + window]
    chunk_video, state = pipe.generate_next_chunk(
        state,
        camera_poses=camera_chunk,
    )

video = pipe.finalize_autoregressive_state(state)
```

`generate_next_chunk` returns the newly finalized video frames plus the next
state. For camera control, the first chunk should provide `window` poses. Later
chunks may either provide `window` new poses, in which case the pipeline
prepends the cached previous boundary pose, or provide `window + 1` poses
including that boundary pose explicitly. For pre-rendered warp conditioning,
initialize with `conditioning_type="warp"` and pass exactly `window` warp frames
per call via `warp_video` and optionally `warp_visibility_mask`.

An interactive browser UI is available for prompt-and-button camera control:

```bash
python scripts/web_control.py \
  --host 0.0.0.0 \
  --port 7860
```

Open the printed URL, upload a first frame, enter a prompt, select translation
and rotation buttons, then click Generate. The server keeps the autoregressive
state alive between Generate clicks. Generated mp4 files are written under `runs/web_control` by default. 

<a href="assets/webcontrol_demo.mp4">
  <img src="assets/webcontrol_demo.gif" alt="WebControl demo" width="100%">
</a>

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

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026warpashistorygeneralizablecameracontrolledvideo,
      title={Warp-as-History: Generalizable Camera-Controlled Video Generation from One Training Video}, 
      author={Yifan Wang and Tong He},
      year={2026},
      eprint={2605.15182},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15182}, 
}
```

## Acknowledgements

We sincerely thank the authors of
[Helios](https://github.com/PKU-YuanGroup/Helios) for releasing such an
excellent open-source video generation model. Warp-as-History is built directly
on top of Helios, and this work would not be possible without their model,
codebase, and open research contribution.

## License

- Helios code and weights follow the upstream Helios license:
  https://github.com/PKU-YuanGroup/Helios
- Pi3X code and weights follow the upstream Pi3 license:
  https://github.com/yyfz/Pi3
- Warp-as-History code authored in this repository is licensed under
  Apache-2.0; see [LICENSE](LICENSE).
- LoRA weights are released under CC BY-NC 4.0 and are strictly
  non-commercial.
- Some training/inference examples are derived from one publicly available video
  sequence from the DAVIS Challenge dataset. The original DAVIS data is not
  covered by this repository license and should be obtained from the official
  DAVIS website: https://davischallenge.org/. Please follow the DAVIS dataset
  terms and cite the corresponding DAVIS papers when using DAVIS-derived data.
