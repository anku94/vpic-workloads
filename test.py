import unittest
from util import VPICReader
from reneg import Rank


class TestVPICReader(unittest.TestCase):
    def setUp(self):
        self.vpicReader = VPICReader('.')
        self.assertEqual(self.vpicReader.get_num_ranks(), 32)
        return

    def teardown(self):
        return


class TestRank(unittest.TestCase):
    def setUp(self):
        self.vpicReader = VPICReader('.')
        self.rank = Rank(self.vpicReader, 2)
        return

    def test_1_insert_init(self):
        self.rank.insert(range(20))
        self.assertEqual(len(self.rank.oobLeft), 20,
                         msg="init insertion not okay")
        self.assertEqual(len(self.rank.oobRight), 0,
                         msg="init insertion not okay")
        self.assertIsNone(self.rank.pivots)
        self.assertIsNone(self.rank.pivotCounts)

    def test_2_rank_pivots(self):
        pivots = self.rank.compute_pivots(5)
        self.assertEqual(pivots, [0, 5, 10, 15, 19])
        return

    def test_3_update_pivots(self):
        new_pivots = [3, 7, 11, 13]
        self.rank.update_pivots(new_pivots)
        self.rank.flush_oobs()
        self.assertEqual(self.rank.pivotCounts, [4, 4, 2])
        self.assertEqual(len(self.rank.oobLeft), 3)
        self.assertEqual(len(self.rank.oobRight), 7)

    def test_4_get_pivots_again(self):
        pivots = self.rank.compute_pivots(3)
        self.assertEqual(pivots, [0, 10, 19])

        pivots = self.rank.compute_pivots(7)
        self.assertEqual(pivots, [0, 3, 7, 10, 13, 17, 19])

    def test_repartition(self):
        pivots_old = [3, 10, 13, 19]
        counts_old = [2, 7, 3]

        pivots_new = [2, 4, 10, 13, 23]
        counts_new = []

        self.rank.__class__.repartition_bin_counts(pivots_old, counts_old,
                                                   pivots_new, counts_new)
        self.assertAlmostEqual(sum(counts_old), sum(counts_new))
        self.assertAlmostEqual(counts_new[0], 0.2857, 3)
        self.assertAlmostEqual(counts_new[3], 3.0, 3)
        return

    def tearDown(self):
        return


if __name__ == '__main__':
    """ yeah yeah we're doing component testing and
    not unit testing calm down """
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
