import json
from pathlib import Path


class BenchmarkRunner:

    def __init__(self):
        self.episodes = []

    def add_episode(
        self,
        task,
        task_success,
        vision_records,
        action_records,
        planner_steps,
        invalid_actions,
        recovery_attempts,
        recovery_successes,
    ):
        episode = {
            "task": task,
            "task_success": bool(task_success),

            "vision_records": vision_records,
            "action_records": action_records,

            "planner_steps": planner_steps,
            "invalid_actions": invalid_actions,

            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
        }

        self.episodes.append(episode)

    def task_success_rate(self):
        if len(self.episodes) == 0:
            return 0.0

        success_count = sum(
            episode["task_success"]
            for episode in self.episodes
        )

        return success_count / len(self.episodes)

    def pick_success_rate(self):
        return self._action_success_rate("pick")

    def place_success_rate(self):
        return self._action_success_rate("place")

    def mean_place_error(self):
        errors = []

        for episode in self.episodes:
            for record in episode["action_records"]:
                if (
                    record.get("action") == "place"
                    and record["position_error"] is not None
                ):
                    errors.append(
                        record["position_error"]
                    )

        if len(errors) == 0:
            return None

        return sum(errors) / len(errors)

    def _action_success_rate(self, action_name):
        action_records = [
            record
            for episode in self.episodes
            for record in episode["action_records"]
            if (
                record.get("action") == action_name
                and record.get("gt_success") is not None
            )
        ]

        if len(action_records) == 0:
            return 0.0

        success_count = sum(
            record["gt_success"]
            for record in action_records
        )

        return success_count / len(action_records)

    def vision_success_rate(self):
        vision_records = []

        for episode in self.episodes:
            vision_records.extend(episode["vision_records"])

        if len(vision_records) == 0:
            return 0.0

        success_count = sum(
            record["success"]
            for record in vision_records
        )

        return success_count / len(vision_records)

    def mean_localization_error(self):
        errors = []

        for episode in self.episodes:
            for record in episode["vision_records"]:
                if record["localization_error"] is not None:
                    errors.append(
                        record["localization_error"]
                    )

        if len(errors) == 0:
            return None

        return sum(errors) / len(errors)

    def invalid_action_rate(self):
        total_steps = sum(
            episode["planner_steps"]
            for episode in self.episodes
        )

        if total_steps == 0:
            return 0.0

        total_invalid = sum(
            episode["invalid_actions"]
            for episode in self.episodes
        )

        return total_invalid / total_steps

    def average_planner_steps(self):
        if len(self.episodes) == 0:
            return 0.0

        total_steps = sum(
            episode["planner_steps"]
            for episode in self.episodes
        )

        return total_steps / len(self.episodes)

    def recovery_success_rate(self):
        total_attempts = sum(
            episode["recovery_attempts"]
            for episode in self.episodes
        )

        if total_attempts == 0:
            return 0.0

        total_successes = sum(
            episode["recovery_successes"]
            for episode in self.episodes
        )

        return total_successes / total_attempts

    def save_results(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        results = {
            "num_episodes": len(self.episodes),

            "summary": {
                "task_success_rate": self.task_success_rate(),
                "vision_success_rate": self.vision_success_rate(),
                "mean_localization_error": self.mean_localization_error(),
                "pick_success_rate": self.pick_success_rate(),
                "place_success_rate": self.place_success_rate(),
                "mean_place_error": self.mean_place_error(),
                "invalid_action_rate": self.invalid_action_rate(),
                "average_planner_steps": self.average_planner_steps(),
                "recovery_success_rate": self.recovery_success_rate(),
            },

            "episodes": self.episodes,
        }

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(path)

