class RecoveryPolicy:

    def __init__(self, max_retries=2):
        self.max_retries = max_retries


    def count_consecutive_errors(self, history, action, error):
        """
        从history最后一条开始，
        统计相同action和error连续出现了多少次。
        """

        count = 0

        for record in reversed(history):

            if (
                record.get("action") == action
                and record.get("error") == error
            ):
                count += 1

            else:
                break

        return count

    def decide(self, state):

        last_result = state.get("last_result")
        last_action = state.get("last_action")
        last_error = state.get("last_error")
        history = state.get("history", [])

        # 上一步没有失败，不需要Recovery
        if last_result is not False:
            return {
                "mode": "continue",
                "reason": None
            }

        retry_count = self.count_consecutive_errors(
            history,
            last_action,
            last_error
        )

        # 连续失败次数超过限制
        if retry_count > self.max_retries:
            return {
                "mode": "abort",
                "reason": last_error
            }

        # Pick阶段视觉失败，可以重新观察再抓
        if (
            last_error == "vision_failed"
            and last_action == "pick"
        ):
            return {
                "mode": "retry",
                "reason": last_error
            }

        # 真正没有抓住，可以重新抓
        if last_error == "pick_failed":
            return {
                "mode": "retry",
                "reason": last_error
            }

        # Place后不能盲目重复place
        if last_error in {
            "place_failed",
            "vision_failed"
        } and last_action == "place":
            return {
                "mode": "replan",
                "reason": last_error
            }

        # 运动控制失败
        if last_error == "motion_failed":
            return {
                "mode": "abort",
                "reason": last_error
            }

        # 状态/Planner产生的不合法动作
        if last_error in {
            "already_holding",
            "not_holding_box",
            "invalid_target",
            "invalid_action"
        }:
            return {
                "mode": "replan",
                "reason": last_error
            }

        # 无法识别的错误，保守停止
        return {
            "mode": "abort",
            "reason": last_error
        }