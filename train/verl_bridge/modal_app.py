"""Run the verl GRPO stages on a Modal GPU instead of a local machine.

    modal run train/verl_bridge/modal_app.py --stage smoke         # 1 step, does it run at all?
    modal run --detach train/verl_bridge/modal_app.py --stage full # keeps going after you close the laptop
    modal shell train/verl_bridge/modal_app.py::train              # live shell in the same image + GPU + volume

This file only packages things. The run itself is still train/verl_bridge/pilot.sh, so a
plain SSH GPU box (RunPod / Lambda / Nebius) runs the exact same command without Modal.

What lives where:
    image   verl + SGLang + CUDA, the Brownout repo (code only), the Cloud-OpsBench clone
    volume  /vol: HF model cache, data/verl parquet, runs/ (trajectories), checkpoints
    secrets HF_TOKEN, WANDB_API_KEY (created once with `modal secret create`, never in code)

Read first:
    Modal's own verl GRPO example (closest thing to this file; copy its image setup):
        https://modal.com/docs/examples  -> search "verl"
        github.com/modal-labs/modal-examples  06_gpu_and_ml/reinforcement-learning/
    Images:   https://modal.com/docs/guide/images      (from_registry, add_local_dir, env)
    Volumes:  https://modal.com/docs/guide/volumes     (commit, `modal volume get/put`)
    Secrets:  https://modal.com/docs/guide/secrets
    GPUs:     https://modal.com/docs/guide/gpu         (gpu="H100", "A100-80GB")
    Timeouts: https://modal.com/docs/guide/timeouts    (max 24 h per call)
    verl install / docker images: https://verl.readthedocs.io -> Installation
"""

import modal

APP_NAME = "brownout-verl"
REPO = "/root/brownout"                 # where the Brownout code sits inside the container
VOL = "/vol"                            # the volume mount
CLOUDOPSBENCH = "/opt/cloudopsbench"    # read-only benchmark data, baked into the image

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("brownout-train", create_if_missing=True)


# ---------------------------------------------------------------- image
# TODO(you): pick the base. Two options, try A first:
#   A) verl's prebuilt docker image with SGLang (tag from the verl install docs):
#        modal.Image.from_registry("<verl sglang image:tag>", add_python=None)
#   B) a CUDA base + pip install verl[sglang] (pin the version, see README step 3).
#      Slower to build, easier to pin exactly.
# Whatever you pick: write the verl + sglang versions into the README / write-up.
image = (
    modal.Image.from_registry("TODO")
    # TODO(you): Cloud-OpsBench into the image (public, MIT). Pin a commit so runs are reproducible:
    #   .run_commands(f"git clone https://github.com/LLM4Ops/Cloud-OpsBench {CLOUDOPSBENCH}",
    #                 f"cd {CLOUDOPSBENCH} && git checkout <commit>")
    # TODO(you): any Brownout deps that verl's image lacks (python-dotenv? pyarrow/pandas?)
    #   .pip_install(...)
    .env({
        "CLOUDOPSBENCH_ROOT": CLOUDOPSBENCH,
        "HF_HOME": f"{VOL}/hf",          # model weights download once, then stay on the volume
        "PYTHONPATH": REPO,
    })
    # The repo goes in LAST so a code edit doesn't rebuild the layers above.
    # TODO(you): check the ignore= argument in the Images guide. Never ship .env, runs/, .git,
    #   raw_call.py. Secrets come from modal.Secret, not from .env.
    .add_local_dir(".", REPO, ignore=["TODO"])
)

secrets = [
    # TODO(you): `modal secret create huggingface HF_TOKEN=...` and `... wandb WANDB_API_KEY=...`
    #   (in your terminal, not here), then:
    # modal.Secret.from_name("huggingface"),
    # modal.Secret.from_name("wandb"),
]


# ---------------------------------------------------------------- train
@app.function(
    image=image,
    gpu="H100",                 # 1 GPU: Qwen3-8B + LoRA. "A100-80GB" is cheaper, slower.
    volumes={VOL: volume},
    secrets=secrets,
    timeout=24 * 60 * 60,       # Modal's maximum; the full run must fit (pilot tells you)
)
def train(stage: str = "smoke"):
    """One pilot.sh stage, reading data from and writing runs/checkpoints to the volume.

    TODO(you), in this order:
      1. Data: make sure {VOL}/data/verl/cloudops-boutique/{train,val}.parquet exist.
         Either build them here (python -m train.verl_bridge.make_dataset ... --out {VOL}/...)
         or upload once from the laptop with `modal volume put`.
      2. Env for pilot.sh: STAGE=stage, MODEL=Qwen/Qwen3-8B, BROWNOUT_TOOL_PARSER=hermes,
         NGPUS=1, DATA={VOL}/data/verl/cloudops-boutique, and make BROWNOUT_SAVE_DIR and
         verl's checkpoint dir (trainer.default_local_dir) point INTO {VOL}, or they vanish
         when the container stops.
      3. subprocess.run(["bash", "train/verl_bridge/pilot.sh", ...], cwd=REPO, env=..., check=True)
      4. volume.commit() at the end (and ideally also if it crashes: try/finally).
    """
    raise NotImplementedError


@app.function(image=image, gpu="A100", volumes={VOL: volume}, secrets=secrets, timeout=30 * 60)
def check():
    """Cheap pre-flight before paying for a real stage (~5 GPU-minutes).

    TODO(you):
      1. The verl-names check script from earlier (AgentLoopOutput fields, AgentLoopMetrics,
         ToolParser names, extract_tool_calls signature, import train.verl_bridge.agent_loop).
      2. python -m train.verl_bridge.check_template Qwen/Qwen3-8B   -> "OK: identical"
      3. python -m unittest tests.token_model_test                  -> 11/11 in the real image
    """
    raise NotImplementedError


@app.local_entrypoint()
def main(stage: str = "smoke", check_only: bool = False):
    # TODO(you): if check_only: check.remote(); else train.remote(stage)
    #   Then pull results to a machine with disk space:
    #   modal volume get brownout-train runs/train-<stage> ./runs/
    raise NotImplementedError
