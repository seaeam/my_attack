import unittest

import numpy as np

from heir import Heirattack


class AttributeNodeBudgetTests(unittest.TestCase):
    def make_attack(self, *, visits, max_visits=15, total_nodes=None):
        attack = Heirattack.__new__(Heirattack)
        attack._attacked_nodes = dict(visits)
        attack.text_attack_max_visits = max_visits
        attack.text_attack_total_nodes = total_nodes
        return attack

    def test_global_budget_counts_distinct_nodes_not_visits(self):
        attack = self.make_attack(visits={1: 2, 2: 15}, total_nodes=3)

        selected = attack.filter_text_attack_nodes([2, 1, 3, 4, 3])

        np.testing.assert_array_equal(selected, np.array([1, 3]))

    def test_existing_nodes_can_be_revisited_after_unique_budget_is_full(self):
        attack = self.make_attack(visits={1: 2, 2: 4}, total_nodes=2)

        selected = attack.filter_text_attack_nodes([3, 2, 1])

        np.testing.assert_array_equal(selected, np.array([2, 1]))

    def test_zero_unique_node_budget_disables_new_attribute_targets(self):
        attack = self.make_attack(visits={}, total_nodes=0)

        selected = attack.filter_text_attack_nodes([1, 2, 3])

        np.testing.assert_array_equal(selected, np.array([], dtype=np.int64))

    def test_unlimited_budget_preserves_rank_order_and_deduplicates(self):
        attack = self.make_attack(visits={}, total_nodes=None)

        selected = attack.filter_text_attack_nodes([5, 3, 5, 4])

        np.testing.assert_array_equal(selected, np.array([5, 3, 4]))


if __name__ == "__main__":
    unittest.main()
