from ...interface_v1 import Task

TASKS = [{"id": "example", "max_steps": 8}]   # TODO: your tasks; thresholds belong here


class TemplateTasks:
    def load(self):
        for spec in TASKS:
            yield Task(id=spec["id"], seed=0, spec=spec)
