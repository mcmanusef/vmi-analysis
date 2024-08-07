import os
import numpy as np
import AnalysisServer
import h5py
def split_where(iterable, predicate):
    i = 0
    for j, item in enumerate(iterable):
        if predicate(item):
            if i < j:
                yield iterable[i:j]
            i = j
    yield iterable[i:]
def process_tdc(tdc_data):
    for tdc_type, c_time, ftime, _ in tdc_data:
        tdc_time = 3.125 * c_time + .260 * ftime
        if tdc_type == 15:
            tdc_type=1
        elif tdc_type == 14:
            tdc_type=3
        elif tdc_type == 10:
            tdc_type=2
        else:
            tdc_type=4
        yield tdc_type, tdc_time

fname=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_08_000000.tpx3"
outname=fname[:-4]+"h5"

with open(fname, mode='rb') as f:
    # Separate file into 8 byte chunks
    file_size=os.path.getsize(fname)//8
    # file_size=10000

    packets=[f.read(8) for _ in range(file_size)]

    # Split packets into chunks where the first 4 bytes are b'TPX3'

    chunks=list(split_where(packets, lambda x: x[0:4]==b'TPX3'))

    # Reference code: [int.from_bytes(p, byteorder="little") - 2 ** 62 for p in packets]
    chunk_ints=[[int.from_bytes(p, byteorder="little") - 2 ** 62 for p in chunk] for chunk in chunks]
    processed_chunks=[AnalysisServer.process_chunk(chunk) for chunk in chunk_ints]
    try:
        pixel_data=np.concatenate([chunk[0] for chunk in processed_chunks if chunk[0]])
        x,y,toa,tot=pixel_data[:,0],pixel_data[:,1],pixel_data[:,2],pixel_data[:,3]
    except ValueError:
        print("No pixel data found")
        x,y,toa,tot=np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    try:
        tdc_data=np.concatenate([chunk[1] for chunk in processed_chunks if chunk[1]])
        tdc_datas=list(zip(*tdc_data))
        tdc_type, tdc_time=zip(*process_tdc(tdc_data))
    except ValueError:
        print("No TDC data found")
        tdc_data=np.zeros((0,4))
        tdc_type, tdc_time=np.zeros(0), np.zeros(0)




    print("All chunks processed")

    with h5py.File(outname, mode='w') as f:
        f.create_dataset("x", data=x, chunks=1000)
        f.create_dataset("y", data=y, chunks=1000)
        f.create_dataset("toa", data=toa, chunks=1000)
        f.create_dataset("tot", data=tot, chunks=1000)
        f.create_dataset("tdc_time", data=tdc_time, chunks=1000)
        f.create_dataset("tdc_type", data=tdc_type, chunks=1000)
        print(f"Data saved to {outname}")
#%%
