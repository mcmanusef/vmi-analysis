from vmi_analysis import serval


def acquire_data(folder: str, duration: int|float, prefix: str = "tpx_", file_length: float = 30,
                      bpc_file: str = r"C:\SoPhy\pixelconfig_20240514.bpc", dacs_file: str = r"C:\SoPhy\pixelconfig_20240514.bpc.dacs", frame_time=1):
    serval_ip='http://localhost:8080'
    assert serval.test_connection(), "Serval server not found"
    serval.load_config_files(bpc_file, dacs_file)

    serval.set_acquisition_parameters(
            destination=folder, prefix=prefix, file_length=file_length, frame_time=duration)
    try:
        serval.start_acquisition(n_files=duration//file_length, block=True)
    finally:
        serval.stop_acquisition()


