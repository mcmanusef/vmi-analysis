# Amount of packets that a 64-bit packet can be distanced
# from the sorted position. For a quad this can be at most 512^2
# hit packets but we use a little more to account for other packet types.
UNSORTED = int(3e5)

BLOCK_N = int(2.2*UNSORTED)
BLOCK_SIZE = 8*BLOCK_N

SORT_WORKERS = 4

# Should be multiple of 64 bits
assert BLOCK_SIZE % 8 == 0

assert BLOCK_N > UNSORTED

# A note on terminology:
# a block is an amount of data that is sorted by a single thread.
# a chunk is multiple blocks, corresponding to the data of all threads.
CHUNK_N = SORT_WORKERS * BLOCK_N

