"""A one-shot Harbor agent for LiveCodeBench: one model call, the reply's last Python block
becomes /app/solution.py, and the task's own tests/test.sh gives the reward.

    harbor run -p lcb/livecodebench -i 1873_d -e docker -o jobs -m Qwen/Qwen3-4B \
      -a src.harbor_agents.one_shot:OneShot --ak api_base=http://localhost:30000/v1

Each call is saved to <trial>/agent/one_shot.json. With --ak replay=<that file> the agent
writes the same answer again with no model, so one answer can be graded in two sandboxes
(local Docker, Prime) and the two rewards compared.

Like Proximus, it runs on the host and touches the sandbox only through Harbor's environment
(upload), so the sandbox needs no network and never sees a key. Keys come from the variable
named by api_key_env, never from --ak (Harbor saves agent options in job files).
"""
import json
import os
import re
import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from openai import AsyncOpenAI

SYSTEM = ("You cannot run tools. Solve the problem and reply with one complete Python program "
          "in a ```python block. It must read stdin and write stdout.")
CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
SOLUTION = "/app/solution.py"


class OneShotOptions(AgentOptions):
    api_base: str | None = None          # any OpenAI-compatible endpoint (SGLang, Miles session)
    api_key_env: str = "OPENAI_API_KEY"  # the NAME of the variable holding the key
    temperature: float = 0.0
    max_tokens: int = 4096
    replay: str | None = None            # a saved one_shot.json: reuse its reply, call no model


def last_code_block(text: str | None) -> str | None:
    blocks = CODE.findall(text or "")
    return blocks[-1] if blocks else None


class OneShot(BaseAgent):
    options_model = OneShotOptions

    def __init__(self, *args, **kwargs):
        # The model name goes to the endpoint as-is (e.g. a served checkpoint's name). Harbor's
        # parsing would look it up in LiteLLM's model list, which it downloads from GitHub.
        kwargs["skip_model_parsing"] = True
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "brownout-one-shot"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment) -> None:
        pass

    async def ask(self, instruction: str) -> dict:
        o = self.options
        if o.replay:
            saved = json.loads(Path(o.replay).read_text(encoding="utf-8"))
            return {**saved, "replayed_from": o.replay}
        client = AsyncOpenAI(base_url=o.api_base, api_key=os.environ.get(o.api_key_env, "EMPTY"))
        r = await client.chat.completions.create(
            model=self.model_name, temperature=o.temperature, max_tokens=o.max_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}])
        usage = r.usage
        return {"model": self.model_name, "reply": r.choices[0].message.content,
                "finish_reason": r.choices[0].finish_reason,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None}

    async def run(self, instruction: str, environment, context) -> None:
        record = await self.ask(instruction)
        record["code"] = last_code_block(record.get("reply"))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "one_shot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if not record.get("replayed_from"):
            context.n_input_tokens = record["input_tokens"]
            context.n_output_tokens = record["output_tokens"]
        if record["code"] is None:       # no program in the reply: nothing written, the tests fail
            return
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "solution.py"
            path.write_text(record["code"], encoding="utf-8")
            await environment.upload_file(path, SOLUTION)
