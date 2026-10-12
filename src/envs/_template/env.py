from ...interface_v1 import Task


class TemplateEnv:
    system_prompt = "TODO: who the agent is, what the tools do, when to call submit()."

    def reset(self, task: Task) -> str:
        self.done = False
        self.steps_used = 0
        # TODO: build the world from task.spec
        return "TODO: the opening message the agent sees"

    def step(self, tool, args):
        if self.done:
            return {"error": "episode is already submitted"}
        self.steps_used += 1
        if tool == "submit":
            self.done = True
            return {"status": "submitted"}
        return {"error": f"unknown tool {tool!r}"}   # TODO: your tools

    def tool_schema(self):
        return [{"type": "function", "function": {
            "name": "submit", "description": "End the episode.",
            "parameters": {"type": "object", "properties": {}, "required": []}}}]

    def snapshot(self):
        # facts about the final state, no pass/fail: the verifier reads only this + the trace
        return {"submitted": self.done, "steps_used": self.steps_used}

    def close(self):
        pass
