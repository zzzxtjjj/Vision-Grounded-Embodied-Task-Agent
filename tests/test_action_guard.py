import unittest

from core.action_guard import ActionGuard


class ActionGuardTests(unittest.TestCase):

    def setUp(self):
        self.guard = ActionGuard()

    def test_pick_without_holding_is_valid(self):
        result = self.guard.validate(
            {"action": "pick", "target": None},
            {"holding": None},
        )
        self.assertEqual(
            result,
            {"valid": True, "error": None},
        )

    def test_pick_while_holding_is_rejected(self):
        result = self.guard.validate(
            {"action": "pick", "target": None},
            {"holding": "box"},
        )
        self.assertEqual(
            result,
            {"valid": False, "error": "already_holding"},
        )

    def test_place_without_holding_is_rejected(self):
        result = self.guard.validate(
            {"action": "place", "target": "center"},
            {"holding": None},
        )
        self.assertEqual(
            result,
            {"valid": False, "error": "not_holding_box"},
        )

    def test_place_while_holding_is_valid(self):
        result = self.guard.validate(
            {"action": "place", "target": "center"},
            {"holding": "box"},
        )
        self.assertEqual(
            result,
            {"valid": True, "error": None},
        )

    def test_unknown_action_is_rejected(self):
        result = self.guard.validate(
            {"action": "move", "target": None},
            {"holding": None},
        )
        self.assertEqual(
            result,
            {"valid": False, "error": "invalid_action"},
        )


if __name__ == "__main__":
    unittest.main()
