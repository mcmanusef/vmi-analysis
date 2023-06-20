import asyncio

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 1234)

    print(f'Send: {message!r}')
    writer.write(message.encode())
    await writer.drain()
    print('Close the connection')
    writer.close()
    await writer.wait_closed()

async def main():
    with open(r"C:\Users\mcman\Code\VMI\Data\kr001_p\kr001_p000000.tpx3", mode='rb') as f:
        reader, writer = await asyncio.open_connection('127.0.0.1', 1234)
        writer.write("DATA".encode())
        await writer.drain()
        for i in range(100000000000000):
            x=f.read(8)
            # print(len(x))
            if len(x)<8:
                break
            # print(x)
            writer.write(x)
            await writer.drain()
    writer.write_eof()
    writer.close()
    await writer.wait_closed()

    # await asyncio.sleep(10)
    # await tcp_echo_client('DUMP')

while True:
    try:
        asyncio.run(main())
    except ConnectionError:
        pass
#%%
