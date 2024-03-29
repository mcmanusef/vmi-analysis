import asyncio
import os
import time
from AnalysisServer import AnalysisServer


async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 1234)

    print(f'Send: {message!r}')
    writer.write(message.encode())
    await writer.drain()
    print('Close the connection')
    writer.close()
    await writer.wait_closed()

async def main(folder,num=10000,skip_first=0):
    reader, writer = await asyncio.open_connection('127.0.0.1', 1234)
    try:
        for fname,i in zip(sorted(os.listdir(folder)), range(num)):
            if i < skip_first:
                continue
            print(fname)
            with open(os.path.join(folder,fname), mode='rb') as f:
                await writer.drain()
                while True:
                    x=f.read(8)
                    if len(x)<8:
                        break
                    writer.write(x)
                    await writer.drain()
    except Exception:
        pass
    finally:
        writer.write_eof()
        writer.close()
        await writer.wait_closed()

async def runserv(name, out_name=None):
    if out_name is None:
        out_name=name+".cv4"
    with AnalysisServer(
            filename=out_name,
            max_size=100000,
            cluster_loops=6,
            processing_loops=6,
            max_clusters=2,
            pulse_time_adjust=-500,
            diagnostic_mode=False,
    ) as aserv:
        task1=asyncio.create_task(aserv.start())
        time.sleep(10)
        task2=asyncio.create_task(main(name,num=20,skip_first=0))
        await task1
        await task2
    # asyncio.run(main(r"C:\Users\mcman\Code\VMI\Data\xe001_p",num=1,skip_first=0))q
    print("Done")

if __name__ == '__main__':
    name=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208\xe_07_b"
    out_name=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240129\o2_04_e_s.cv4"
    asyncio.run(runserv(name))
#%%
