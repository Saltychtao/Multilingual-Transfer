from tasks.nli import NLITask


def build_task(task):
    if task == "nli":
        return NLITask()