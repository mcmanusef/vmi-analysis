import asyncio
import os
import time
import argparse
from indev.AnalysisServer import AnalysisServer


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

async def runserv(name):
    with AnalysisServer(
            filename=name+".cv4",
            max_size=100000,
            cluster_loops=6,
            processing_loops=6,
            max_clusters=2,
            pulse_time_adjust=-500,
            diagnostic_mode=False,
    ) as aserv:
        task1=asyncio.create_task(aserv.start())
        input()
        task2=asyncio.create_task(main(name,num=1000,skip_first=0))
        await task1
        await task2
    # asyncio.run(main(r"C:\Users\mcman\Code\VMI\Data\xe001_p",num=1,skip_first=0))q
    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cv4 converter', description="converts .cv4 files to .cv3")
    parser.add_argument("filename")
    args=parser.parse_args()
    name=args.filename
    asyncio.run(runserv(name))
#%%
