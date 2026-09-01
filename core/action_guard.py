class ActionGuard:

    VALID_ACTIONS = {
        "pick",
        "place",
        "finish"
    }

    VALID_PLACE_TARGETS = {
        "left",
        "center",
        "right"
    }

    def validate(self, action, state):

        action_type = action.get("action")
        target = action.get("target")
        holding = state.get("holding")

        # 1. action类型是否合法
        if action_type not in self.VALID_ACTIONS:
            return {
                "valid": False,
                "error": "invalid_action"
            }

        # 2. Pick的前置条件
        if action_type == "pick":

            if holding is not None:
                return {
                    "valid": False,
                    "error": "already_holding"
                }

            return {
                "valid": True,
                "error": None
            }

        # 3. Place的前置条件
        if action_type == "place":

            if holding != "box":
                return {
                    "valid": False,
                    "error": "not_holding_box"
                }

            if target not in self.VALID_PLACE_TARGETS:
                return {
                    "valid": False,
                    "error": "invalid_target"
                }

            return {
                "valid": True,
                "error": None
            }

        # 4. finish暂时允许
        return {
            "valid": True,
            "error": None
        }