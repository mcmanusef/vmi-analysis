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

async def main(folder,num=10000,skip_first=0):
    reader, writer = await asyncio.open_connection('127.0.0.1', 1234)
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

    writer.write_eof()
    writer.close()
    await writer.wait_closed()

if __name__ == '__main__':
    asyncio.run(main(r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230913\air_s",num=1000,skip_first=1))
    # asyncio.run(main(r"C:\Users\mcman\Code\VMI\Data\xe001_p",num=1,skip_first=0))
    print("Done")
#%%
