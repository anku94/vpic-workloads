This codebase simulates a `Renegotiation` consisting of some n `Ranks`, or writing processes.

First, some utility classes.

1. `utils.VPICReader` - This is a wrapper to read VPIC traces. These traces can then be passed to the simulator (each read just gets you an array of floats for each rank).
2. `utils.Histogram` - This is a simple Histogram object, but one cool feeature. This can be thought of as a simple wrapper over np.histogram, but with a `rebalance(nbins: int)` function. `rebalance` redivides the histogram into `nbins` buckets with equal mass, irrespective of the number of buckets or masses the Histogram initially had. It uses linear interpolation to arrive at these equal-mass buckets.

The two main classes are:

1. `rank.Rank` - this simulates the state for an individual rank. Each rank has a pivots object which stores the currently negotiated rank-partition boundaries. oob_left and oob_right are the out-of-bounds buffers, these store data that the rank is asked to write, but does not fall in any of the bin boundaries. Each rank has a `compute_pivots` function that returns n samples that are representative of its distribution, and contain equal 'mass' between them. The mass is also one of the arguments returned by this function.

2. `reneg.Renegtiation` - This wraps an array of `Rank` objects - and provides a global object for the simulation. It manipulates the states of each rank depending on whether you want to read/write data, or renegotiate.o

The tests are very barebones - and are just there to try out some functionality of some module. The driver code is in `reneg_bench.py`.

`reneg_bench.py` generates different graphs to observe different properties of the whole system.

- `benchmark_range_accuracy`, for example, checks how accurately the pivot computation scheme describes the data.

- `benchmark_predictive_power_suite` renegotiates at every 5%/1% intervals, and looks at load-balancing properties of the data. These graphs are stored in vis/.

The trace we're running right now has 32 ranks and 4 timestep dumps. (in code, ts indices 0, 1, 2, 3 correspond to timesteps 100, 950, 1900, 2850 in the actual VPIC simulation.)
