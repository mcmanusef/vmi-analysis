import time

from vmi_analysis import serval
import os


def acquire_data(folder: str, duration: int | float,
                 prefix: str = "tpx_",
                 bpc_file: str = r"C:\SoPhy\pixelconfig_20240514.bpc",
                 dacs_file: str = r"C:\SoPhy\pixelconfig_20240514.bpc.dacs",
                 frame_time: int | float = 10):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # assert serval.test_connection(), "Serval server not found"
    if serval_busy():
        serval.stop_acquisition()
        time.sleep(1)

    serval.load_config_files(bpc_file, dacs_file)

    serval.set_acquisition_parameters(
            destination=folder, prefix=prefix, duration=duration, frame_time=frame_time)
    try:
        serval.start_acquisition(block=False)
    except Exception as e:
        serval.stop_acquisition()
        raise e


def serval_busy():
    # assert serval.test_connection(), "Serval server not found"
    return serval.get_dash()['Measurement']['Status'] != "DA_IDLE"


def stop_acquisition():
    # assert serval.test_connection(), "Serval server not found"
    serval.stop_acquisition()


def get_status():
    # assert serval.test_connection(), "Serval server not found"
    return serval.get_dash()['Measurement']['Status']


def get_dash():
    # assert serval.test_connection(), "Serval server not found"
    return serval.get_dash()
