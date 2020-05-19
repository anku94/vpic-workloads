import unittest
from util import VPICReader
from reneg import Renegotiation


class TestRank(unittest.TestCase):
    reneg = None

    def setUp(self) -> None:
        pass

    def test_1(self) -> None:
        # for num_pivots in [4, 8, 16, 32, 64, 128, 256]:
        #     self._test_num_pivots(num_pivots)
        # pass
        self._test_num_pivots(256)

    def _test_num_pivots(self, num_pivots):
        vpic_reader = VPICReader('../data')
        reneg = Renegotiation(32, 0, vpic_reader)
        reneg.set_NUM_PIVOTS(num_pivots)
        reneg.read_all()
        reneg.insert(0.05)
        reneg.plot()
        del vpic_reader
        del reneg
        vpic_reader = None
        reneg = None

    def tearDown(self) -> None:
        pass
