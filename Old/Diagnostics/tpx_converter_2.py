import os
import numpy as np
from Old import AnalysisServer
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
        tdc_time = 3.125 * c_time + 0.260 * ftime
        if tdc_type == 15:
            tdc_type = 1
        elif tdc_type == 14:
            tdc_type = 3
        elif tdc_type == 10:
            tdc_type = 2
        else:
            tdc_type = 4
        yield tdc_type, tdc_time

def process_file(fname):
    with open(fname, mode='rb') as f:
        # Separate file into 8-byte chunks
        file_size = os.path.getsize(fname) // 8
        packets = [f.read(8) for _ in range(file_size)]

        # Split packets into chunks where the first 4 bytes are b'TPX3'
        chunks = list(split_where(packets, lambda x: x[0:4] == b'TPX3'))

        # Process each chunk
        chunk_ints = [
            [int.from_bytes(p, byteorder="little") - 2 ** 62 for p in chunk]
            for chunk in chunks
        ]
        processed_chunks = [AnalysisServer.process_chunk(chunk) for chunk in chunk_ints]

        x, y, toa, tot = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        tdc_time, tdc_type = np.zeros(0), np.zeros(0)

        try:
            pixel_data = np.concatenate([chunk[0] for chunk in processed_chunks if chunk[0]])
            x, y, toa, tot = pixel_data[:, 0], pixel_data[:, 1], pixel_data[:, 2]*25/16, pixel_data[:, 3]
        except ValueError:
            print(f"No pixel data found in {fname}")

        try:
            tdc_data = np.concatenate([chunk[1] for chunk in processed_chunks if chunk[1]])
            tdc_type, tdc_time = zip(*process_tdc(tdc_data))
            tdc_type = np.array(tdc_type)
            tdc_time = np.array(tdc_time)
        except ValueError:
            print(f"No TDC data found in {fname}")

        print(f"All chunks processed for {fname}")
        return x, y, toa, tot, tdc_time, tdc_type

# Specify the file or directory path
path = r"D:\Data\xe002_s"
end=10
if os.path.isfile(path):
    x, y, toa, tot, tdc_time, tdc_type = process_file(path)
    # Define output filename
    base, _ = os.path.splitext(path)
    outname = base + ".h5"
    # Write data to HDF5 file
    with h5py.File(outname, mode='w') as h5f:
        h5f.create_dataset("x", data=x, chunks=(1000,))
        h5f.create_dataset("y", data=y, chunks=(1000,))
        h5f.create_dataset("toa", data=toa, chunks=(1000,))
        h5f.create_dataset("tot", data=tot, chunks=(1000,))
        h5f.create_dataset("tdc_time", data=tdc_time, chunks=(1000,))
        h5f.create_dataset("tdc_type", data=tdc_type, chunks=(1000,))
        print(f"Data saved to {outname}")
elif os.path.isdir(path):
    # Process all .tpx and .tpx3 files in the directory and combine data
    tpx_files = [
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(('.tpx', '.tpx3'))
    ]

    # Initialize lists to accumulate data
    x_list = []
    y_list = []
    toa_list = []
    tot_list = []
    tdc_time_list = []
    tdc_type_list = []

    for tpx_file in tpx_files[:end]:
        x, y, toa, tot, tdc_time, tdc_type = process_file(tpx_file)
        if x.size > 0:
            x_list.append(x)
            y_list.append(y)
            toa_list.append(toa)
            tot_list.append(tot)
        if tdc_time.size > 0:
            tdc_time_list.append(tdc_time)
            tdc_type_list.append(tdc_type)

    # Concatenate all data
    x_all = np.concatenate(x_list) if x_list else np.zeros(0)
    y_all = np.concatenate(y_list) if y_list else np.zeros(0)
    toa_all = np.concatenate(toa_list) if toa_list else np.zeros(0)
    tot_all = np.concatenate(tot_list) if tot_list else np.zeros(0)
    tdc_time_all = np.concatenate(tdc_time_list) if tdc_time_list else np.zeros(0)
    tdc_type_all = np.concatenate(tdc_type_list) if tdc_type_list else np.zeros(0)

    # Define output filename
    outname = path+".h5"
    # Write combined data to HDF5 file
    with h5py.File(outname, mode='w') as h5f:
        h5f.create_dataset("x", data=x_all, chunks=(1000,))
        h5f.create_dataset("y", data=y_all, chunks=(1000,))
        h5f.create_dataset("toa", data=toa_all, chunks=(1000,))
        h5f.create_dataset("tot", data=tot_all, chunks=(1000,))
        h5f.create_dataset("tdc_time", data=tdc_time_all, chunks=(1000,))
        h5f.create_dataset("tdc_type", data=tdc_type_all, chunks=(1000,))
        print(f"Combined data saved to {outname}")
else:
    print(f"{path} is not a valid file or directory.")

#%%
