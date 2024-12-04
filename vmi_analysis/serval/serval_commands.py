import time

import requests
import json
import pathlib

DEFAULT_IP= 'http://localhost:8080'


def set_acquisition_parameters(destination, prefix, file_length, frame_time, serval_ip=DEFAULT_IP):
    config=json.loads(requests.get(serval_ip + '/detector/config').text)
    config['BiasVoltage']=100
    config['TriggerMode']='CONTINUOUS'
    config['ExposureTime']=config['TriggerPeriod']=frame_time
    config['nTriggers']= file_length//frame_time
    resp=requests.put(serval_ip + '/detector/config', data=json.dumps(config))
    assert resp.status_code == 200, f"Error setting acquisition parameters: {resp.text}"

    destination = {
        "Raw": [{"Base": pathlib.Path(destination).as_uri(),
                 "FilePattern": prefix,
                 }],
    }

    resp=requests.put(serval_ip + '/server/destination', data=json.dumps(destination))
    assert resp.status_code == 200, f"Error setting destination: {resp.text}"


def load_config_files(bpc_file, dacs_file, serval_ip=DEFAULT_IP):
    resp= requests.get(serval_ip + '/config/load?format=pixelconfig&file=' + bpc_file)
    assert resp.status_code == 200, f"Error loading bpc file: {resp.text}"
    resp= requests.get(serval_ip + '/config/load?format=dacs&file=' + dacs_file)
    assert resp.status_code == 200, f"Error loading dacs file: {resp.text}"


def test_connection(serval_ip=DEFAULT_IP):
    try:
        requests.get(serval_ip)
        dashboard = requests.get(serval_ip + '/dashboard')
        json.loads(dashboard.text)
        return True
    except requests.exceptions.ConnectionError:
        return False


def start_acquisition(num_files=1, serval_ip=DEFAULT_IP, block=True):
    if not block:
        raise NotImplementedError("Non-blocking acquisition not implemented")

    for i in range(num_files):
        resp=requests.get(serval_ip+'/measurement/start')
        assert resp.status_code == 200, f"Error starting acquisition {i}: {resp.text}"

        while json.loads(requests.get(serval_ip+'/dashboard').text)['Measurement']['Status'] != "DA_IDLE":
            time.sleep(0.1)


def stop_acquisition(serval_ip=DEFAULT_IP):
    resp=requests.get(serval_ip+'/measurement/stop')
    assert resp.status_code == 200, f"Error stopping acquisition: {resp.text}"