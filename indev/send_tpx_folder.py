import argparse
import asyncio
import os


async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 1234)

    print(f'Send: {message!r}')
    writer.write(message.encode())
    await writer.drain()
    print('Close the connection')
    writer.close()
    await writer.wait_closed()

async def main(folder, port):
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    for file in os.listdir(folder):
        with open(file, mode='rb') as f:
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
