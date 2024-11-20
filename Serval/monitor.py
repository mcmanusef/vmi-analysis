import json
import pathlib
import sys
import requests
import processing.pipeline as pipeline


def dict_to_string_recursive(d):
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(k + ':')
            lines.extend(['  ' + l for l in dict_to_string_recursive(v)])
        else:
            lines.append(k + ': ' + str(v))
    return lines

if __name__ == '__main__':
    server = 'http://localhost:8080'
    bpc_file = r"C:\SoPhy\pixelconfig_20240514.bpc"
    dacs_file = r"C:\SoPhy\pixelconfig_20240514.bpc.dacs"
    dest = 'C:/monitor'

    frame_time = 1
    n_frames = sys.maxsize

    resp = requests.get(server + '/config/load?format=pixelconfig&file=' + bpc_file)
    print(resp.text)
    resp = requests.get(server + '/config/load?format=dacs&file=' + dacs_file)
    print(resp.text)

    config = json.loads(requests.get(server + '/detector/config').text)
    print('\n'.join(dict_to_string_recursive(config)))
    config['BiasVoltage'] = 100
    config['TriggerMode'] = 'CONTINUOUS'
    config['ExposureTime'] = config['TriggerPeriod'] = frame_time
    config['nTriggers'] = n_frames
    resp = requests.put(server + '/detector/config', data=json.dumps(config))
    print(resp.text)
    destination = {
        "Raw": [{"Base": pathlib.Path(dest).as_uri(),
                 "FilePattern": "raw%Hms_",
                 }],
    }

    resp = requests.put(server + '/server/destination', data=json.dumps(destination))
    resp = requests.get(server + '/server/destination')
    print(resp.text)
    try:
        resp = requests.get(server + '/measurement/start')
        print(resp.text)

        with pipeline.MonitorPipeline(dest, cluster_processes=4) as mon:
            mon.start()
            mon.wait_for_completion()
    finally:
        resp = requests.get(server + '/measurement/stop')
        print(resp.text)
