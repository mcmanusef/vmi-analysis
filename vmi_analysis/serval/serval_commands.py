import time

import requests
import json
import pathlib

DEFAULT_IP= 'http://localhost:8080'


def set_acquisition_parameters(destination, prefix, duration, frame_time, serval_ip=DEFAULT_IP):
    config=json.loads(requests.get(serval_ip + '/detector/config').text)
    config['BiasVoltage']=100
    config['TriggerMode']='CONTINUOUS'
    config['ExposureTime'] = config['TriggerPeriod'] = frame_time
    config['nTriggers']= int(duration/frame_time)
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
    resp=requests.get(serval_ip+'/dashboard')
    return resp.status_code == 200


def start_acquisition(serval_ip=DEFAULT_IP, block=True):
    resp=requests.get(serval_ip+'/measurement/start')
    assert resp.status_code == 200, f"Error starting acquisition {i}: {resp.text}"
    if block:
        while is_busy(serval_ip):
            time.sleep(0.1)


def stop_acquisition(serval_ip=DEFAULT_IP):
    resp=requests.get(serval_ip+'/measurement/stop')

def is_busy(serval_ip=DEFAULT_IP):
    return json.loads(requests.get(serval_ip+'/dashboard').text)['Measurement']['Status'] != "DA_IDLE"