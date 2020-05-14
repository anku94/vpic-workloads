import sys

from util import VPICReader
from reneg import Rank, Renegotiation

def run():
    vpicReader = VPICReader(data_path)
    rank0 = Rank(vpicReader, 0)
    print(rank0.get_id())
    rank0.read(0)
    #reneg = Renegotiation(None, None)
    #  print(reneg)

if __name__ == '__main__':
    assert(len(sys.argv) == 2)

    data_path = sys.argv[1]
    run()
