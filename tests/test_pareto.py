import unittest

from droidblend.pareto import ProfilePoint, pareto_frontier, select_fastest_within_budget
from droidblend.schedules import LayerGroup


def point(group, quality, latency):
    return ProfilePoint(group, quality, 1.0, "qa_f1", latency, 100.0, 1, None, 10)


class ParetoTests(unittest.TestCase):
    def test_selects_fastest_feasible_point(self):
        points = [point(LayerGroup(0, 1), 0.8, 10), point(LayerGroup(1, 3), 0.97, 25), point(LayerGroup(0, 4), 1.0, 50)]
        selected = select_fastest_within_budget(points, 0.05)
        self.assertEqual(selected.group, LayerGroup(1, 3))
        self.assertEqual([item.group for item in pareto_frontier(points, 0.05)], [LayerGroup(1, 3), LayerGroup(0, 4)])


if __name__ == "__main__":
    unittest.main()
