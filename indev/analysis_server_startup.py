from indev.AnalysisServer import AnalysisServer
import multiprocessing
import asyncio
async def start(analysis_server, port=1234):
    for loop in analysis_server.get_loops():
        loop_process = multiprocessing.Process(target=loop)
        loop_process.start()
    print("Loops Started")

    server = await asyncio.start_server(analysis_server.make_connection_handler(),'127.0.0.1',port=port)
    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')
    async with server:
        await server.serve_forever()

if __name__=="__main__":
    aserv=AnalysisServer(max_size=10000,cluster_loops=8,processing_loops=4)
    asyncio.run(start(aserv))
#%%
