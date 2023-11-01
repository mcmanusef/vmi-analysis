import argparse
import os
import shutil
import h5py

def main(file):
    out_name=file[:-1]+'3'
    print(out_name)

    assert file[-1]=='4'

    shutil.copyfile(file, out_name)

    with h5py.File(out_name, mode='r+') as f:
        times=[
            f['t'],
            f['t_etof'],
            f['t_tof'],
        ]
        for td in times:
            td[...]=td[()]*1000


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='cv4 converter', description="converts .cv4 files to .cv3")
    parser.add_argument("filename")
    args=parser.parse_args()

    if args.filename[-4:]=='.cv4':
        main(args.filename)
    else:
        for file in os.listdir(args.filename):
            print(file)
            if file[-4:]=='.cv4':
                main(os.path.join(args.filename, file))

