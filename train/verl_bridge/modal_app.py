"""Run pilot.sh on a Modal GPU. Everything that must survive the container goes in /vol.
Model and GPU are set once below (MODEL, GPU).

    modal run train/verl_bridge/modal_app.py::download         # CPU only: MODEL into /vol (once per model)
    modal run train/verl_bridge/modal_app.py::check            # build the image + pre-flight checks
    modal run --detach train/verl_bridge/modal_app.py::train --stage smoke
    modal app logs brownout-verl                               # plain-text logs (no spinner redraws)

Mid-run, without stopping it (the volume is committed every 2 min):
    modal volume get brownout-train logs/smoke.log runs/smoke.log        # verl output
    modal volume get brownout-train logs/smoke.gpu.log runs/smoke.gpu.log  # GPU mem/util per minute

Once:  modal secret create brownout HF_TOKEN=... WANDB_API_KEY=...
After: modal volume ls brownout-train runs        # one folder per run: train-<model>-<stage>-<MMDD-HHMM>
       then `pack` it (see pack() below: folder downloads don't work on Windows)

Setup follows Modal's Miles GRPO example (.agents/skills/modal/references/examples/miles_grpo.md).
"""

import os
import shutil
import subprocess
import threading
import time

import modal

app = modal.App("brownout-verl")
vol = modal.Volume.from_name("brownout-train", create_if_missing=True)

# The one switch for model size. Qwen3-4B fits an H100; Qwen3-8B needs an H200 (141 GB):
# on 80 GB its fp32 trainer copy (33 GB) + SGLang + the first weight sync run out of memory.
MODEL = "Qwen/Qwen3-4B"
GPU = "H100"
NAME = MODEL.split("/")[-1].lower()                         # "qwen3-4b": runs/checkpoints per model
# Task categories left out of TRAINING (val/test keep all). The pilot showed these give only
# all-0 GRPO groups for Qwen3-4B (no gradient, ~20% of compute). "" = train on everything.
EXCLUDE_TRAIN = "boutique-admission,boutique-codedefect,boutique-performance"
MODEL_DIR = f"/vol/models/{MODEL.split('/')[-1]}"

image = (
    # verl's prebuilt SGLang stack (torch, sglang, flash-attn, ray). It does NOT include verl itself.
    # sgl0512 = SGLang 0.5.12, the version verl v0.9.1 pins (sgl055 was too old for it)
    modal.Image.from_registry("verlai/verl:sgl0512.dev4")
    .entrypoint([])                                         # the image's own entrypoint can block Modal's
    .run_commands(
        # verl pinned (latest release). --no-deps keeps the image's torch/sglang, so verl's one
        # extra runtime dep, TransferQueue (needed by the default V1 trainer), is installed by hand.
        "pip install --no-deps git+https://github.com/volcengine/verl.git@v0.9.1",
        # pinned: TQ main (after v0.1.11) raises on fields a GRPO run never writes
        # (ref_log_prob, values); verl v0.9.1 was released against v0.1.10/0.1.11
        "pip install --no-deps git+https://github.com/Ascend/TransferQueue.git@v0.1.11",
        "git clone https://github.com/LLM4Ops/Cloud-OpsBench /opt/cloudopsbench",
    )
    .env({
        "PYTHONUNBUFFERED": "1",            # print immediately; piped Python output is otherwise held back
        "NCCL_NVLS_ENABLE": "0",
        # trainer and SGLang share the GPU and pass weights via CUDA IPC; expandable segments
        # need a syscall Modal's sandbox doesn't allow (pidfd_getfd)
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    })
    .add_local_dir(".", "/root/brownout", ignore=[".env", ".git", "runs", "raw_call.py"])
)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(secrets=[modal.Secret.from_name("brownout")])
def environment():
    os.getenv("HF_TOKEN")
    os.getenv("WANDB_API_KEY")


@app.function(image=download_image, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("brownout")], timeout=30 * 60)
def download():
    """Weights onto the volume on a cheap CPU container, so the H100 never waits on a download."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, local_dir=MODEL_DIR)           # TODO: pin revision= for the write-up
    vol.commit()
    print("downloaded to", MODEL_DIR)


@app.function(image=image, gpu="H100", volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("brownout")], timeout=30 * 60)
def check():
    """Pre-flight, instead of `modal shell` (not on Windows). Stops at the first failure."""
    os.chdir("/root/brownout")
    env = {**os.environ, "CLOUDOPSBENCH_ROOT": "/opt/cloudopsbench", "HF_HOME": "/vol/hf"}
    for cmd in [
        "nvidia-smi",
        "python -c \"import verl, sglang; print('verl', verl.__version__, 'sglang', sglang.__version__)\"",
        "python -c \"import train.verl_bridge.agent_loop; print('bridge imports ok')\"",
        # by file, not `tests.token_model_test`: some pip package in the image ships its own
        # top-level `tests` package, which shadows our (namespace) tests/ folder
        "python -m unittest discover -s tests -p token_model_test.py",
        f"python -m train.verl_bridge.check_template {MODEL}",
    ]:
        print(f"\n$ {cmd}", flush=True)
        subprocess.run(cmd, shell=True, env=env, check=True)
    vol.commit()                                            # keep the downloaded tokenizer


@app.function(volumes={"/vol": vol}, timeout=10 * 60)
def pack(folder: str):
    """Tar a runs/ folder into one file, because `modal volume get <dir>` on Windows writes every
    file to the same local path (you get one file, the last one).

        modal run train/verl_bridge/modal_app.py::pack --folder runs/train-qwen3-4b-smoke
        modal volume get brownout-train runs/train-qwen3-4b-smoke.tar.gz runs/ --force
        tar -xzf runs/train-qwen3-4b-smoke.tar.gz -C runs
    """
    import tarfile

    src = f"/vol/{folder.strip('/')}"
    if not os.path.isdir(src):
        runs = sorted(d for d in os.listdir("/vol/runs") if os.path.isdir(f"/vol/runs/{d}"))
        raise SystemExit(f"no folder {src}. Run folders on the volume:\n  "
                         + "\n  ".join(f"runs/{d}" for d in runs))
    out = f"{src}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(src, arcname=os.path.basename(src))
    vol.commit()
    n = sum(len(files) for _, _, files in os.walk(src))
    print(f"{n} files -> {out} ({os.path.getsize(out) // 1024} KiB)")


def _copy_episodes(local_dir: str, vol_dir: str) -> int:
    """Episodes are written to the container's own disk (plain files, written by the Ray
    workers), then copied onto the volume here. Returns how many there are."""
    if not os.path.isdir(local_dir):
        return 0
    shutil.copytree(local_dir, vol_dir, dirs_exist_ok=True)
    return sum(len(files) for _, _, files in os.walk(local_dir))   # train/ + val/


def _watch(stage: str, stop: threading.Event, local_dir: str, vol_dir: str):
    """Every minute: GPU memory/util into logs/<stage>.gpu.log. Every 2 min: copy episodes to
    the volume and commit, so logs and episodes can be pulled with `modal volume get` mid-run."""
    n = 0
    while not stop.wait(60):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        with open(f"/vol/logs/{stage}.gpu.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {gpu}\n")
        n += 1
        if n % 2 == 0:
            # `ray status` lists "Pending demands": a silent hang waiting on CPUs/GPUs shows up here
            ray = subprocess.run(["ray", "status"], capture_output=True, text=True)
            with open(f"/vol/logs/{stage}.ray.log", "a") as f:
                f.write(f"===== {time.strftime('%H:%M:%S')}\n{ray.stdout}{ray.stderr}\n")
            _copy_episodes(local_dir, vol_dir)
            vol.commit()


@app.function(image=image, gpu=GPU,
              cpu=8, memory=98304,  # default is 0.125 cores / 128 MiB: far too little for Ray + SGLang.
              volumes={"/vol": vol},  # 96 GiB RAM: the trainer's fp32 model copy is offloaded here
              # 6 h cap: the 50-step run needs ~4 h. A silent hang (error 9 in DEBUGGING.md) can't
              # bill more than this; full resumes from its last 10-step checkpoint if cut off
              secrets=[modal.Secret.from_name("brownout")], timeout=6 * 3600)
def train(stage: str = "smoke"):
    os.chdir("/root/brownout")
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        raise SystemExit(f"no model at {MODEL_DIR}: run `modal run ...modal_app.py::download` first")
    data_dir = f"/root/data/{stage}"                        # local; rebuilt below, nothing to keep
    env = {**os.environ,
           "STAGE": stage,
           "MODEL": MODEL_DIR,                              # local copy, no download on the GPU
           "BROWNOUT_TOOL_PARSER": "hermes",
           "BROWNOUT_MODEL_ID": f"{NAME}-lora-{stage}",
           "EXPERIMENT": f"cloudops-grpo-{NAME}-{stage}",  # W&B run name
           "NGPUS": "1",
           "CLOUDOPSBENCH_ROOT": "/opt/cloudopsbench",
           "HF_HOME": "/vol/hf",
           "DATA": data_dir,
           # our agent loop saves every episode here (local disk); _copy_episodes puts them
           # on the volume at runs/train-<model>-<stage>
           "BROWNOUT_SAVE_DIR": f"/root/episodes/train-{NAME}-{stage}",
           "HYDRA_FULL_ERROR": "1"}                         # full traceback, not the short one

    # rebuilt every run (seconds), so EXCLUDE_TRAIN changes always take effect
    subprocess.run(["python", "-m", "train.verl_bridge.make_dataset",
                    "--split-file", "splits/cloudops-boutique.json", "--out", data_dir,
                    "--exclude-train", EXCLUDE_TRAIN], env=env, check=True)

    os.makedirs("/vol/logs", exist_ok=True)
    # What Ray sees, written at the top of the log. If GPU is missing or CPU is tiny here,
    # verl waits forever without an error.
    subprocess.run(["bash", "-c", "python -c \"import os, ray; ray.init(); "
                    "print('cpu_count:', os.cpu_count(), 'ray resources:', ray.cluster_resources(), "
                    f"flush=True); ray.shutdown()\" 2>&1 | tee /vol/logs/{stage}.log"],
                   env=env, check=True)

    local_dir = env["BROWNOUT_SAVE_DIR"]
    # one folder per run, so episodes of different runs never mix (e.g. 1016-0912 = Oct 16, 09:12 UTC)
    vol_dir = f"/vol/runs/train-{NAME}-{stage}-{time.strftime('%m%d-%H%M')}"
    print(f"episodes will be copied to {vol_dir}", flush=True)
    stop = threading.Event()
    threading.Thread(target=_watch, args=(stage, stop, local_dir, vol_dir), daemon=True).start()
    try:
        subprocess.run(["bash", "-o", "pipefail", "-c",
                        f"bash train/verl_bridge/pilot.sh trainer.default_local_dir=/vol/checkpoints/{NAME}-{stage}"
                        f" 2>&1 | tee -a /vol/logs/{stage}.log"],
                       env=env, check=True)
    finally:
        stop.set()
        n = _copy_episodes(local_dir, vol_dir)
        print(f"episodes saved: {n}  ({local_dir} -> {vol_dir})", flush=True)
        vol.commit()                                        # keep runs/ even if it crashed


# ------------------------------------------------------------------ evaluation
# verl's own validation, alone (pilot.sh STAGE=eval): the exact episode path that gave the
# in-training val scores, so base and a checkpoint are compared like for like. (The API path,
# src.run against an SGLang server, drops the model's earlier <think> text; Qwen3-4B then
# loops on plain-text answers and never calls Submit. See DEBUGGING.md.)
#
#   modal run --detach train/verl_bridge/modal_app.py::evaluate --step 40 --split val   # sanity: ~0.40
#   modal run --detach train/verl_bridge/modal_app.py::evaluate --step 0  --split test  # base
#   modal run --detach train/verl_bridge/modal_app.py::evaluate --step 40 --split test
#   modal run --detach train/verl_bridge/modal_app.py::evaluate --step 0  --split all   # every task
#   then pack runs/eval-qwen3-4b-<step>-<split>-<MMDD-HHMM> and run_report on it

@app.function(image=image, gpu=GPU, cpu=8, memory=98304, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("brownout")], timeout=2 * 3600)
def evaluate(step: int = 0, split: str = "test", samples: int = 1):
    """step: 0 = the base model, N = checkpoints/<NAME>-full/global_step_N.
    samples: 1 = greedy (as the in-training val); >1 = that many samples per task at the
    training temperature (mean@N: less noise, N times the episodes)."""
    os.chdir("/root/brownout")
    ckpt = f"/vol/checkpoints/{NAME}-full/global_step_{step}"
    if step and not os.path.isdir(ckpt):
        raise SystemExit(f"no checkpoint {ckpt}")
    label = f"eval-{NAME}-{'base' if step == 0 else f'step{step}'}-{split}"
    data_dir = "/root/data/eval"
    env = {**os.environ,
           "STAGE": "eval",
           "MODEL": MODEL_DIR,
           "BROWNOUT_TOOL_PARSER": "hermes",
           "BROWNOUT_MODEL_ID": label,
           "EXPERIMENT": label,                             # W&B run name
           "NGPUS": "1",
           "CLOUDOPSBENCH_ROOT": "/opt/cloudopsbench",
           "HF_HOME": "/vol/hf",
           "DATA": data_dir,
           # all = every Boutique task (upstream's table). Episodes still land in train/ val/ test/
           "VAL_FILE": (f"[{data_dir}/train.parquet,{data_dir}/val.parquet,{data_dir}/test.parquet]"
                        if split == "all" else f"{data_dir}/{split}.parquet"),
           "RESUME": "resume_path" if step else "disable",
           "BROWNOUT_SAVE_DIR": f"/root/episodes/{label}",
           "HYDRA_FULL_ERROR": "1"}
    subprocess.run(["python", "-m", "train.verl_bridge.make_dataset",
                    "--split-file", "splits/cloudops-boutique.json", "--out", data_dir],
                   env=env, check=True)

    # val_batch_size: verl otherwise sends the whole val set at once. 116 at once is known to
    # work; 464 at once hung (DEBUGGING.md Error 18)
    overrides = ["trainer.default_local_dir=/root/checkpoints", "data.val_batch_size=116"]
    if step:
        overrides.append(f"trainer.resume_from_path={ckpt}")
    if samples > 1:
        overrides += [f"actor_rollout_ref.rollout.val_kwargs.n={samples}",
                      "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
                      "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
                      "actor_rollout_ref.rollout.val_kwargs.top_p=0.95"]

    os.makedirs("/vol/logs", exist_ok=True)
    log = f"/vol/logs/{label}.log"
    local_dir = env["BROWNOUT_SAVE_DIR"]
    vol_dir = f"/vol/runs/{label}-{time.strftime('%m%d-%H%M')}"
    print(f"episodes will be copied to {vol_dir}", flush=True)
    stop = threading.Event()
    threading.Thread(target=_watch, args=(label, stop, local_dir, vol_dir), daemon=True).start()
    try:
        subprocess.run(["bash", "-o", "pipefail", "-c",
                        f"bash train/verl_bridge/pilot.sh {' '.join(overrides)} 2>&1 | tee {log}"],
                       env=env, check=True)
    finally:
        stop.set()
        n = _copy_episodes(local_dir, vol_dir)
        print(f"episodes saved: {n}  ({local_dir} -> {vol_dir})", flush=True)
        vol.commit()
    # the scores: verl prints them once, in the step:0 metrics line
    subprocess.run(["bash", "-c", f"grep -o 'val-[a-z]*/[^ ]*:[0-9.]*' {log} | sort -u"])
