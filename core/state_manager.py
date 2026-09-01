import copy


class RobotStateManager:

    def __init__(self):
        self.state = {
            "holding": None,
            "last_action": None,
            "last_result": None,
            "last_target": None,
            "last_error": None,
            "history": []
        }


    def update_holding(self, value):
        self.state["holding"] = value


    def update_action(self, action):
        """
        保存最近一次执行的动作

        例如:
        "pick"
        "place"
        """

        self.state["last_action"] = action

    def update_result(self, result):
        """
        保存动作结果

        例如:
        True
        False
        """
        self.state["last_result"] = result

    def update_target(self, target):
        """
        保存最近一次 place 的目标

        target:
            left
            right
            center
        """
        self.state["last_target"] = target

    def update_error(self, error):
        self.state["last_error"] = error

    def add_history(self, action, result, error=None):
        record = {
            "action": action,
            "result": result,
            "error": error
        }
        self.state["history"].append(record)

    def reset_task(self):
        """
        新任务开始时清理任务记忆

        注意:
        holding 不清理

        因为 holding 是跨任务持续的 Runtime 状态信念，
        不是 MuJoCo Ground Truth。
        Benchmark Episode reset 会在已知初始状态下单独清理它。
        """
        self.state["last_action"] = None
        self.state["last_result"] = None
        self.state["last_error"] = None
        self.state["last_target"] = None
        self.state["history"] = []

    def get_state(self):
        return copy.deepcopy(self.state)

    def get_error(self):
        return self.state["last_error"]
