import unittest

from droidblend.schedules import LayerGroup, enumerate_layer_groups, transition_layers


class ScheduleTests(unittest.TestCase):
    def test_all_groups_are_continuous_and_legal(self):
        groups = enumerate_layer_groups(4, "all")
        self.assertEqual(len(groups), 10)
        self.assertIn(LayerGroup(0, 4), groups)
        self.assertIn(LayerGroup(3, 4), groups)
        self.assertEqual(transition_layers(groups), [0, 1, 2, 3])

    def test_constraints_are_applied(self):
        groups = enumerate_layer_groups(8, [2, 3], start_layers=[1, 4])
        self.assertEqual(groups, [LayerGroup(1, 3), LayerGroup(4, 6), LayerGroup(1, 4), LayerGroup(4, 7)])


if __name__ == "__main__":
    unittest.main()
