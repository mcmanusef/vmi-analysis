import argparse
import shutil
import h5py

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='cv4 converter', description="converts .cv4 files to .cv3")
    parser.add_argument("filename")
    args=parser.parse_args()
    out_name=args.filename[:-1]+'3'

    assert args.filename[-1]=='4'

    shutil.copyfile(args.filename, out_name)

    with h5py.File(out_name, mode='r+') as f:
        times=[
        f['toa'],
        f['t_etof'],
        f['t_tof'],
            ]
        for td in times:
            td[...]=td[()]*1000

