import argparse
import asyncio
import os
async def main(folder, port):
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    for file in sorted(os.listdir(folder)):
        print(file)
        with open(os.path.join(folder,file), mode='rb') as f:
            while True:
                x=f.read(8)
                if len(x)<8:
                    break
                writer.write(x)
                await writer.drain()
    writer.write_eof()
    writer.close()
    await writer.wait_closed()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='send', description="sends data from a tpx3 file to a local analysis server")

    parser.add_argument('--port', dest='port',default=1234, type=int,
                        help="port of the analysis server")

    parser.add_argument('path')

    args = parser.parse_args()

    asyncio.run(main(args.path, port=args.port))
#%%
