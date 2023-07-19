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
    reader, writer = await asyncio.open_connection('127.0.0.1', 1234)
    for fname in (f"n2_{i:06d}.tpx3" for i in range(18)):
        print(fname)
        with open(fr"J:\ctgroup\DATA\UCONN\VMI\VMI\20230717\n2_s_photodiode\{fname}", mode='rb') as f:
            # reader, writer = await asyncio.open_connection('192.168.93.6', 1234)
            await writer.drain()
            while True:
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
#
# while True:
#     try:
asyncio.run(main())
    #     break
    # except ConnectionError:
    #     pass
print("Done")
#%%
