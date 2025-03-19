import os
import pathlib
import time

import requests
from attrs import define
from cattrs import structure

DEFAULT_IP = "http://localhost:8080"


class ServalException(Exception):
    pass


def set_acquisition_parameters(
    destination,
    duration: int | float = 999999,
    frame_time: int | float = 1,
    prefix="tpx",
    serval_ip=DEFAULT_IP,
):
    resp = requests.get(serval_ip + "/detector/config")
    if resp.status_code != 200:
        raise ServalException(
            f"Error getting current acquisition parameters: {resp.text}"
        )
    config = resp.json()
    config["BiasVoltage"] = 100
    config["TriggerMode"] = "CONTINUOUS"
    config["ExposureTime"] = config["TriggerPeriod"] = frame_time
    config["nTriggers"] = int(duration / frame_time)
    resp = requests.put(serval_ip + "/detector/config", json=config)
    if resp.status_code != 200:
        raise ServalException(f"Error setting acquisition parameters: {resp.text}")

    if not isinstance(destination, dict):
        if (
            os.path.exists(destination)
            or os.path.sep in destination
            or destination.startswith(".")
        ):
            destination = {
                "Raw": [
                    {
                        "Base": pathlib.Path(destination).as_uri(),
                        "FilePattern": prefix,
                    }
                ],
            }
        else:
            destination = {
                "Raw": [
                    {
                        "Base": destination,
                        "FilePattern": prefix,
                    }
                ],
            }

    resp = requests.put(serval_ip + "/server/destination", json=destination)
    if resp.status_code != 200:
        raise ServalException(f"Error setting destination: {resp.text}")


def load_config_files(bpc_file, dacs_file, serval_ip=DEFAULT_IP):
    resp = requests.get(serval_ip + "/config/load?format=pixelconfig&file=" + bpc_file)
    if resp.status_code != 200:
        raise ServalException(f"Error loading bpc file: {resp.text}")

    resp = requests.get(serval_ip + "/config/load?format=dacs&file=" + dacs_file)
    if resp.status_code != 200:
        raise ServalException(f"Error loading dacs file: {resp.text}")


def test_connection(serval_ip=DEFAULT_IP):
    resp = requests.get(serval_ip + "/dashboard")
    return resp.status_code == 200


def start_acquisition(serval_ip=DEFAULT_IP, block=True, force_restart=True):
    resp = requests.get(serval_ip + "/measurement/start")
    if force_restart:
        stop_acquisition(serval_ip)
        while get_status(serval_ip) != "DA_IDLE":
            time.sleep(0.1)

    resp = requests.get(serval_ip + "/measurement/start")
    if resp.status_code != 200:
        raise ServalException(f"Error starting acquisition: {resp.text}")
    if block:
        while get_status(serval_ip) != "DA_IDLE":
            time.sleep(0.1)


def stop_acquisition(serval_ip=DEFAULT_IP):
    resp = requests.get(serval_ip + "/measurement/stop")
    if resp.status_code != 200:
        raise ServalException(f"Error stopping acquisition: {resp.text}")


@define
class Dashboard:
    @define
    class Server:
        Notifications: list[str] | None
        SoftwareVersion: str | None
        SoftwareTimestamp: str | None

    @define
    class Measurement:
        StartDateTime: int | None
        TimeLeft: int | None
        ElapsedTime: float | None
        FrameCount: int | None
        PixelEventRate: int | None
        Status: str | None

    @define
    class Detector:
        DetectorType: str | None

    Server: Server | None
    Measurement: Measurement | None
    Detector: Detector | None


def get_dash(serval_ip=DEFAULT_IP) -> Dashboard:
    resp = requests.get(serval_ip + "/dashboard")
    if resp.status_code != 200:
        raise ServalException(f"Error getting dashboard: {resp.text}")
    return structure(resp.json(), Dashboard)


def get_status(serval_ip=DEFAULT_IP):
    return get_dash(serval_ip).Measurement.Status
