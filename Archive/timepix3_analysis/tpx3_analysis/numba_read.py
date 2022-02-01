import sys
import time

import numpy as np
from numba import njit
import numba as nb

# These constants are configurations for the conversion.
# They need to be module globals, so that Numba interprets
# them as constants which significantly speeds up the conversion.
# The downside is that we need a recompilation every time these constants
# change. See 'compileFunctions' at the end of this file.

chipIndexToOrientation = None
chipIndexToOrigin = None

maxX = None
maxY = None

clusterEnabled = True

maxTOADiff = None
clusterSquareSize = None

def setClusterEnabled(enabled):
    global clusterEnabled
    
    clusterEnabled = enabled

def setMaxTOADiff(diff):
    global maxTOADiff

    maxTOADiff = diff

def setClusterSquareSize(size):
    global clusterSquareSize

    clusterSquareSize = size

def setBoardLayout(layout):
    global chipIndexToOrientation, chipIndexToOrigin, maxX, maxY
     
    maxY = max([l['Y'] for l in layout]) 
    maxX = max([l['X'] for l in layout]) 
     
    chipIndexToOrientation = np.zeros((len(layout), 6), dtype=np.int8)
    chipIndexToOrigin = np.zeros((len(layout), 2), dtype=np.uint16)
    
    height = 256*(maxY + 1)
      
    for i, l in enumerate(layout):
        chipIndexToOrigin[i][0] = 256*l['X']
        chipIndexToOrigin[i][1] = height - ((l['Y']+1) * 256)
        
        orientation = l['Orientation']
        
        assert orientation in ['LtRBtT', 'RtLBtT', 'LtRTtB', 'RtLTtB', 'BtTLtR', 'TtBLtR', 'BtTRtL', 'TtBRtL']
         
        if orientation == 'LtRBtT':
            # x, 255 - y
            chipIndexToOrientation[i] = (0, 1, 0,   1, 0, -1)
        elif orientation == 'TtBLtR':
            # y, x
            chipIndexToOrientation[i] = (0, 0, 1,   0, 1, 0)
        elif orientation == 'RtLTtB':
            # 255 - x, y
            chipIndexToOrientation[i] = (1, -1, 0,   0, 0, 1)
        elif orientation == 'BtTLtR':
            # y, 255 - x
            chipIndexToOrientation[i] = (0, 0, 1,   1, -1, 0)
        elif orientation == 'RtLBtT':
            # 255 - x, 255 - y
            chipIndexToOrientation[i] = (1, -1, 0,   1, 0, -1)
        elif orientation == 'TtBRtL':
            # 255 - y, x
            chipIndexToOrientation[i] = (1, 0, -1,   0, 1, 0)
        elif orientation == 'LtRTtB':
            # x, y
            chipIndexToOrientation[i] = (0, 1, 0,   0, 0, 1)
        elif orientation == 'BtTRtL':
            # 255 - y, 255 - x
            chipIndexToOrientation[i] = (1, 0, -1,   1, -1, 0)
    
@nb.njit(cache=True, nogil=True)
def reorder(d1, d2, d3, indicices):
    d1c = np.copy(d1)
    d2c = np.copy(d2)
    d3c = np.copy(d3)

    for i, idx in enumerate(indicices):
        d1[i] = d1c[idx]
        d2[i] = d2c[idx]
        d3[i] = d3c[idx]

@njit(inline='always')
def getBits(data, high, low) -> nb.uint64:
    num = (high - low) + 1
    mask: nb.uint64 = (1 << num) - 1 # Trick: 2**N - 1 gives N consecutive ones
    maskShifted = mask << low
     
    return (data & maskShifted) >> low

@njit(inline='always')
def calculateXY(data: nb.uint64, chipIndex: nb.uint8) -> (nb.uint16, nb.uint16):
    encoded = data >> 44
    # doublecolumn * 2
    dcol = (encoded & 0x0FE00) >> 8
    # superpixel * 4
    spix = (encoded & 0x001F8) >> 1 # (16+28+3-2)
    # pixel
    pix = (encoded & 0x00007)
    x, y = (dcol + pix // 4), (spix + (pix & 0x3))

    assert chipIndex <= len(chipIndexToOrigin), "Chip index too large for board layout, please use a different board layout"
    assert chipIndex <= len(chipIndexToOrientation), "Chip index too large for board layout, please use a different board layout"
     
    A, B, C, D, E, F = chipIndexToOrientation[chipIndex]
    coords = A*255 + B*x + C*y, D*255 + E*x + F*y
     
    x0, y0 = chipIndexToOrigin[chipIndex]
    return x0 + coords[0], y0 + coords[1]


@njit(inline='always')
def matchesNibble(data, nibble) -> nb.boolean:
    return (data >> 60) == nibble

@njit(inline='always')
def getTDCClock(tdc) -> nb.uint64:
    # Notice we take one bit less than we could (34 bits instead of 35 bits)
    # this is to keep the rollover correction exactly the same for TDC and hits.
    tdcCoarse = (tdc >> 9) & 0x3ffffffff     
     
    # fractional counts, values 1-12, 0.26 ns
    fract = (tdc >> 5) & 0xf
      
    # Bug: fract is sometimes 0 but it should be 1 <= fract <= 12
    #assert 1 <= fract <= 12, "Incorrect fractional TDC part, is the firmware outdated?"
    
    count = (tdcCoarse*12 + fract - 1) # In counts of ~260ps
    
    # Try to get as close as possible to real value. Note that
    # we only go through a floating point value for the 'correction' part.
    return count*260 + round(0.4166666666666*count)

@njit(inline='always')
def getTOAClock(data) -> nb.uint64:
    # ftoa is on a 640 MHz clock
    # toa is on a 40 MHz clock
    # not sure why we have to multiply coarse by 2^14, taken from Serval code
    ftoa = getBits(data, 19, 16)
    toa = getBits(data, 43, 30)
    coarse = getBits(data, 15, 0)
    
    count = ((((coarse << 14) + toa) << 4) - ftoa) # In multiples of 1.5625 ns
    return count*1562 + count//2

@njit(inline='always')
def getTOTClock(data) -> np.uint16:
    return getBits(data, 29, 20)

@njit(inline='always')
def toaHighBitTransition(prev, current) -> nb.boolean:
    return (prev >> 33) == 0 and (current >> 33) == 1

@njit(inline='always')
def toaPenultimateBitTransition(prev, current) -> nb.boolean:
    # Whether the bit just before the MSB 
    # has transitioned
    return ((prev >> 32) & 0b1) == 0 and ((current >> 32) & 0b1) == 1


@njit(cache=True, nogil=True)
def createSortCriteria(data, prevClock: nb.uint32,
                             extraTime: nb.uint32,
                             upperHalf: nb.boolean,
                             chipIndex: nb.uint8):

    sortCriteria = np.zeros_like(data, dtype=np.uint64)
    chips = np.zeros_like(data, dtype=np.uint8)
    
    tpxHeader = 861425748 # int.from_bytes(b'TPX3', 'little')
     
    for (j, d) in enumerate(data):
        isHit = matchesNibble(d, 0xb)
        isTDC = matchesNibble(d, 0x6)
        
        if isHit or isTDC:
           
            if isHit: 
                chips[j] = chipIndex
                clock = getTOAClock(d)
            else:
                clock = getTDCClock(d)
             
            halfPassed = prevClock <= (1 << 44) and clock > (1 << 44) # Passed 17.6s mark
            belowQuarter = clock < (1 << 43) # Quarter mark we define as ~8.8s
            quartedPassed = prevClock <= (1 << 43) and not belowQuarter
             
            # We get close to a rollover if we are in upperhalf of 26.8s time span.
            # If this is the case all TOA which are below quarter time must be a result
            # of rollover and get an extra 26.8s added.
            upperHalf |= halfPassed
            
            # If TOA is full we need to add 34 bits of 1.5625 ns.
            # Written to minimize floating point issues:
            twentySix = ((1 << 34) - 1)*1562 + ((1 << 34) - 1)//2              
             
            # If we pass the quarter mark we 'flush' the correction.
            # All hits from now on get the extra bits as we are unlikely to find a
            # hit from the previous 'cycle'. 
            extraTime += twentySix * (upperHalf and quartedPassed)
            upperHalf &= not (upperHalf and quartedPassed) # Reset

            prevClock = clock
            sortCriteria[j] = clock + extraTime + (belowQuarter and upperHalf)*twentySix
        elif (d & ((1 << 32) - 1) == tpxHeader):
            chipIndex = getBits(d, 39, 32)
        elif (d >> 56) & 0xFFEF == 0x71A0:
            # End of readout command, only coarse time is available
            sortCriteria[j] = getBits(d, 15, 0) << 18
        else:
            # Doesn't matter where the packet ends up
            # use the previous value to keep the data nearly sorted
            # which will make the sorting algorithm perform better.
            # Note that if j == 0 then we select the last element (-1) which is zero
            sortCriteria[j] = sortCriteria[j - 1]

    return sortCriteria, chips, prevClock, extraTime, upperHalf, chipIndex

class SortCriteriaGenerator:
    
    def __init__(self):
        self.state = (0, 0, 0, 0)
    
    def process(self, data):
        criteria, chips, *self.state = createSortCriteria(data, *self.state)
        return criteria, chips

@nb.njit(nogil=True, inline='always')
def getClusterIndex(x: nb.uint16, y: nb.uint16, toa: nb.uint64,
    # State:
    clusterSquares, clusterToa, clusterCount: nb.uint64) -> (nb.uint32, nb.uint64):
          
    xIdx = (x // clusterSquareSize) + 1
    yIdx = (y // clusterSquareSize) + 1
     
    for i in [4, 0, 1, 2, 3, 5, 6, 7, 8]:
        xNeighbour = xIdx + (i % 3) - 1
        yNeighbour = yIdx + i//3 - 1
         
        cToa = clusterToa[yNeighbour, xNeighbour]
        
        if toa - cToa < maxTOADiff:
            
            cIdx = clusterSquares[yNeighbour, xNeighbour]
             
            if cIdx != -1:
                return (cIdx, clusterCount)
     
    newIndex = clusterCount
    clusterSquares[yIdx, xIdx] = newIndex
    clusterToa[yIdx, xIdx] = toa
    clusterCount += 1
    
    return (newIndex, clusterCount)

@nb.njit(cache=False, nogil=True, fastmath=True)
def unpack(data, sortCriteria, chipsSorted, gaps,
    # State:
    frameIndex, lastTdc, lastFrame,
    # Cluster state:
    clusterSquares, clusterToa, clusterCount):
     
    assert len(data) == len(sortCriteria)
    
    maxSize = data.size
    
    tdcType = np.empty(maxSize, dtype=np.uint8)
    tdcTime = np.empty(maxSize, dtype=np.int64)
    
    chips = np.empty(maxSize, dtype=np.uint8)
    x = np.empty(maxSize, dtype=np.uint16)
    y = np.empty(maxSize, dtype=np.uint16)
    tot = np.empty(maxSize, dtype=np.uint16)
    toa = np.empty(maxSize, dtype=np.int64)
    frame = np.empty(maxSize, dtype=np.uint32)
    
    if clusterEnabled:
        clusterIndex = np.empty(maxSize, dtype=np.uint32)
        
    tpxHeader = 861425748 # int.from_bytes(b'TPX3', 'little')
    
    hitIndex = 0
    tdcIndex = 0
    
    for (i, d) in enumerate(data):
        
        if matchesNibble(d, 0xb):
            chips[hitIndex] = chipsSorted[i]

            x_, y_ = calculateXY(d, chipsSorted[i])
            x[hitIndex], y[hitIndex] = x_ + (x_//256)*gaps, y_ + (y_//256)*gaps
            
            tot[hitIndex] = getTOTClock(d)
            frame[hitIndex] = frameIndex

            toa_ = sortCriteria[i]
            toa[hitIndex] = toa_
            
            if clusterEnabled:
                clusterIndex[hitIndex], clusterCount = getClusterIndex(x_, y_, toa_, clusterSquares, clusterToa, clusterCount)
             
            hitIndex += 1
        elif matchesNibble(d, 0x6) and d != lastTdc:
            lastTdc = d
            
            if d >> 56 == 0x6f:
                type_ = 1
            elif d >> 56 == 0x6a:
                type_ = 2
            elif d >> 56 == 0x6e:
                type_ = 3
            elif d >> 56 == 0x6b:
                type_ = 4
            
            tdcType[tdcIndex] = type_
            tdcTime[tdcIndex] = sortCriteria[i]
            tdcIndex += 1
        
        elif (d >> 48) & 0xFFEF == 0x71A0 and d & 0xFFFF != lastFrame:
            # Todo: maybe use sortCriteria instead?
            lastFrame = d & 0xFFFF
            frameIndex += 1
     
    return tdcType, tdcTime, x, y, tot, toa, frame, \
            clusterIndex if clusterEnabled else None, tdcIndex, hitIndex, \
            frameIndex, lastTdc, lastFrame, clusterCount # State


class NumbaUnpacker:

    def __init__(self, gaps):
        self.frameIndex = 0
        self.lastTdc = 0
        self.lastFrame = 0
        self.clusterCount = 0

        self.gaps = gaps
        
        assert 256 % clusterSquareSize == 0, "Cluster square size should be a power of 2 (has to divide 256)"
         
        SPC = 256 // clusterSquareSize # Squares per chip
        
        # Add extra square at the borders (+3). This saves
        # an if statement in the function 'getClusterIndex' which checks
        # whether the index is within the sensor area.
        self.clusterSquares = np.full(( (maxY+3)*SPC, (maxX+3)*SPC ), -1, np.int32)
        self.clusterToa = np.zeros(( (maxY+3)*SPC, (maxX+3)*SPC ), np.uint64)
     
    def currentFrameIndex(self):
        return self.frameIndex
    
    def unpackSortedData(self, data, sortCriteria, chips):
    
        assert data.dtype == np.uint64 and sortCriteria.dtype == np.uint64 and chips.dtype == np.uint8
        
        tdcType, tdcTime, x, y, tot, toa, frame, clusterIndex, tdcIndex, hitIndex, \
            self.frameIndex, self.lastTdc, self.lastFrame, self.clusterCount = \
                unpack(data, sortCriteria, chips, self.gaps,
                       self.frameIndex, self.lastTdc, self.lastFrame,
                       self.clusterSquares, self.clusterToa, self.clusterCount)
         
        result = {
            'tdc_type': tdcType[:tdcIndex],
            'tdc_time': tdcTime[:tdcIndex],
            
            'x': x[:hitIndex],
            'y': y[:hitIndex],
            'tot': tot[:hitIndex],
            'toa': toa[:hitIndex],
            'frame_number': frame[:hitIndex],
        }

        if clusterEnabled:
            result['cluster_index'] = clusterIndex[:hitIndex]

        return result

def compileFunctions():
    global Unpacker
    # Call the functions with the correct type to trigger the Numba
    # compilation

    unpack.recompile()
     
    createSortCriteria(np.zeros(1, dtype=np.uint64), 0 ,0 ,0, 0) 
    unpacker = NumbaUnpacker(2)
    unpacker.unpackSortedData(np.zeros(1, dtype=np.uint64),np.zeros(1, dtype=np.uint64), np.zeros(1, dtype=np.uint8))
    
    reorder(np.zeros(1, dtype=np.uint64),
            np.zeros(1, dtype=np.uint8),
            np.zeros(1, dtype=np.uint64), np.zeros(1, dtype=np.int64))



