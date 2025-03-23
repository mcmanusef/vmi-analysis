import time

from cattrs import unstructure

from vmi_analysis import serval
import os


def acquire_data(
    folder: str,
    duration: int | float,
    prefix: str = "tpx_",
    bpc_file: str = r"C:\SoPhy\pixelconfig_20241206.bpc",
    dacs_file: str = r"C:\SoPhy\pixelconfig_20240514.bpc.dacs",
    test_dir: str = r"C:\serval_test",
    frame_time: int | float = 10,
):

    if not os.path.exists(folder):
        os.makedirs(folder)
    serval.set_acquisition_parameters(test_dir, 0.1, frame_time=0.1)
    serval.start_acquisition()
    # assert serval.test_connection(), "Serval server not found"
    if serval_busy():
        serval.stop_acquisition()
        time.sleep(1)

    num_files = len(os.listdir(folder))

    serval.load_config_files(bpc_file, dacs_file)

    serval.set_acquisition_parameters(
        destination=folder,
        prefix=prefix + f"{num_files}_",
        duration=duration,
        frame_time=frame_time,
    )
    try:
        serval.start_acquisition(block=False)
    except Exception as e:
        serval.stop_acquisition()
        raise e


def serval_busy() -> bool:
    return serval.get_status() != "DA_IDLE"


def stop_acquisition():
    serval.stop_acquisition()


def get_status() -> str | None:
    return serval.get_status()


def get_dash() -> dict:
    return unstructure(serval.get_dash())
