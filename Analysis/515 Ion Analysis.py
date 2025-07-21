import os

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import skimage

file = r"J:\ctgroup\Edward\DATA\VMI\20250408\calibrated\po_400mW_lin_calibrated.h5"


def process_file(file):
    df = pd.read_hdf(file, key='data')

    p_max = 0.6
    df_filtered = df.query(
            f'(px > -{p_max}) & (px < {p_max}) & (py > -{p_max}) & (py < {p_max}) & (pz > -{p_max}) & (pz < {p_max})')
    df_filtered = df_filtered[(df_filtered['m/q'] > 0) & (df_filtered['m/q'] < 100)]
    gate_names = ('C2H4', 'C2H3O', 'Parent')
    gates = ((20, 25.5), (25, 28), (28, 33))

    def gate(mq):
        for gate_name, (low, high) in zip(gate_names, gates):
            if low <= mq <= high:
                return gate_name
        return "Other"

    df_filtered['gate'] = df_filtered['m/q'].apply(gate)
    df_filtered.sort_values(by=['m/q'], ascending=False, inplace=True)

    df_filtered['pr'] = np.sqrt(df_filtered['px'] ** 2 + df_filtered['py'] ** 2 + df_filtered['pz'] ** 2)
    hist, xe, ye = np.histogram2d(
            df_filtered['m/q'], df_filtered['pr'], bins=(256, 256), range=((0, 65), (0, 0.5))
    )
    xe, ye = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    image = skimage.filters.median(hist.T, skimage.morphology.disk(5))
    f = px.imshow(np.log(image), x=xe, y=ye, aspect='auto', origin='lower',
                  color_continuous_scale='Inferno', title=f'm/q vs pr (log scale), {os.path.split(file)[-1]}',
                  height=900, width=900,
                  labels={'x': 'm/q', 'y': 'pr'},

                  )
    plotly.io.write_image(f, file.replace('_calibrated.h5', '_mq_vs_pr.png'), scale=2)


if __name__ == "__main__":

    path = r"J:\ctgroup\Edward\DATA\VMI\20250408\calibrated"
    for file in os.listdir(path):
        if file.endswith('_calibrated.h5'):
            process_file(os.path.join(path, file))
