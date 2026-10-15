"""Run pilot.sh on a Modal H100. Everything that must survive the container goes in /vol.

    modal run train/verl_bridge/modal_app.py::train --stage smoke
    modal run --detach train/verl_bridge/modal_app.py::train --stage full
    modal shell train/verl_bridge/modal_app.py::train          # debug by hand

Once:  modal secret create brownout HF_TOKEN=... WANDB_API_KEY=...
After: modal volume get brownout-train runs ./runs
"""

import os
import subprocess

import modal

app = modal.App("brownout-verl")
vol = modal.Volume.from_name("brownout-train", create_if_missing=True)

image = (
    # verl's prebuilt SGLang stack (torch, sglang, flash-attn). It does NOT include verl itself.
    modal.Image.from_registry("verlai/verl:sgl055.latest")
    .run_commands(
        # TODO: pin a release tag instead of main (github.com/volcengine/verl/releases), write it down
        "pip install --no-deps git+https://github.com/volcengine/verl.git@main",
        "git clone https://github.com/LLM4Ops/Cloud-OpsBench /opt/cloudopsbench",
    )
    .add_local_dir(".", "/root/brownout", ignore=[".env", ".git", "runs", "raw_call.py"])
)


@app.function(image=image, gpu="H100", volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("brownout")], timeout=24 * 3600)
def train(stage: str = "smoke"):
    os.chdir("/root/brownout")
    env = {**os.environ,
           "STAGE": stage,
           "MODEL": "Qwen/Qwen3-8B",
           "BROWNOUT_TOOL_PARSER": "hermes",
           "BROWNOUT_MODEL_ID": f"qwen3-8b-lora-{stage}",
           "NGPUS": "1",
           "CLOUDOPSBENCH_ROOT": "/opt/cloudopsbench",
           "HF_HOME": "/vol/hf",                            # weights download once
           "DATA": "/vol/data",
           "BROWNOUT_SAVE_DIR": f"/vol/runs/train-{stage}"}

    if not os.path.exists("/vol/data/train.parquet"):
        subprocess.run(["python", "-m", "train.verl_bridge.make_dataset",
                        "--split-file", "splits/cloudops-boutique.json", "--out", "/vol/data"],
                       env=env, check=True)
    try:
        subprocess.run(["bash", "train/verl_bridge/pilot.sh",
                        f"trainer.default_local_dir=/vol/checkpoints/{stage}"],
                       env=env, check=True)
    finally:
        vol.commit()                                        # keep runs/ even if it crashed
