import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io


def main(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    df = pd.read_hdf(file, key='data')
    p_max = 0.6
    df_filtered = df.query(
            f'(px > -{p_max}) & (px < {p_max}) & (py > -{p_max}) & (py < {p_max}) & (pz > -{p_max}) & (pz < {p_max})')
    df_filtered = df_filtered[(df_filtered['m/q'] > 0) & (df_filtered['m/q'] < 100)]
    # & (df_filtered['pz'] > -0.05) & (df_filtered['pz'] < 0.05)
    # & (df_filtered['py'] > -0.05) & (df_filtered['py'] < 0.05)]
    gate_names = ('C2H4', 'C2H3O')
    gates = ((20, 33), (33, 50))

    def gate(mq):
        for gate_name, (low, high) in zip(gate_names, gates):
            if low <= mq <= high:
                return gate_name
        return "Other"

    df_filtered['gate'] = df_filtered['m/q'].apply(gate)
    df_filtered.sort_values(by=['m/q'], ascending=False, inplace=True)
    df_filtered['pr'] = np.sqrt(df_filtered['px'] ** 2 + df_filtered['py'] ** 2 + df_filtered['pz'] ** 2)

    # px.density_heatmap(df_filtered[df_filtered['pr'] < 0.45],
    #                    y='pr', x='m/q', height=900, width=900, nbinsy=256, marginal_y='histogram', marginal_x='histogram',
    #                    nbinsx=256, title='m/q vs pr', range_x=(0, 65), range_y=(0, 0.45), range_color=(0,80),).show()
    hist, xe, ye = np.histogram2d(
            df_filtered['m/q'], df_filtered['pr'], bins=(256, 256), range=((0, 65), (0, 0.5))
    )
    xe, ye = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    xx, yy = np.meshgrid(xe, ye)

    import skimage

    image = hist.T / yy ** 2

    image = skimage.filters.median(image, skimage.morphology.disk(5))
    fig = px.imshow(np.nan_to_num(np.log(image), neginf=0), x=xe, y=ye, aspect='auto', origin='lower',
                    color_continuous_scale='Inferno', title=f'm/q vs pr (log scale), {os.path.split(file)[-1]}',
                    height=900, width=900,
                    labels={'x': 'm/q', 'y': 'pr'},

                    )
    plotly.io.write_image(fig, file.replace('.h5', '.png'), scale=4)


if __name__ == '__main__':
    folder = r"J:\ctgroup\Edward\DATA\VMI\20250303\Propylene Oxide 2W\calibrated"

    for file in os.listdir(folder):
        if file.endswith('.h5'):
            try:
                print(f"Processing {file}...")
                main(os.path.join(folder, file))
                print(f"Finished {file}.")
            except AttributeError as e:
                print(f"Error processing {file}: {e}")
