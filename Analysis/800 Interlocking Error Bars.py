import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import skimage
from scipy.ndimage import rotate
from scipy.signal import argrelmin
from sklearn.cluster import DBSCAN


def bump(x):
    return np.where(np.abs(x) < 1, np.exp(1 / (x ** 2 - 1) + 1) ** 0.2, 0)


def argrelmin_rotated(arr, angle, axis=0, order=1,
                      reshape=True, interp_order=1,
                      mode='constant', cval=np.inf):
    """
    Find relative minima in a 2D array along lines at a given angle
    by rotating the data, using scipy.signal.argrelmin, then mapping
    those minima back to the original array coordinates.

    Parameters
    ----------
    arr : array_like, shape (M, N)
        Your input 2D signal.
    angle : float
        Angle in degrees, measured counter‑clockwise from the +x axis (columns).
    axis : {0, 1}, optional
        Like scipy.signal.argrelmin: axis=0 finds minima down columns
        (i.e. along lines at `angle` in the original), axis=1 along rows
        (lines at `angle + 90°`).
    order : int, optional
        How many neighbors on each side to demand being larger (default 1).
    reshape : bool, optional
        Passed to scipy.ndimage.rotate.  If True, the rotated image
        is large enough to contain the entire original.
    interp_order : int, optional
        The spline‐interpolation order for the rotation (0=nearest, 1=bilinear, …).
        Nearest (0) preserves your exact pixel values.
    mode, cval : optional
        Passed to scipy.ndimage.rotate for areas outside the original image.
        We default to constant `+∞` so no bogus minima appear on the padding.

    Returns
    -------
    minima_mask : ndarray of bool, shape (M, N)
        True in the original array where a strict relative minimum
        was found along the specified angled direction.
    """
    arr = np.asarray(arr)
    M, N = arr.shape

    # rotate by –angle so that lines at +angle become vertical (axis=0)
    arr_rot = rotate(arr, -angle,
                     reshape=reshape,
                     order=interp_order,
                     mode=mode,
                     cval=cval)
    M_rot, N_rot = arr_rot.shape

    # find relative minima in the rotated frame
    rows_rot, cols_rot = argrelmin(arr_rot, axis=axis, order=order)

    # compute centers of original and rotated images
    center_orig = np.array([(M - 1) / 2.0, (N - 1) / 2.0])
    center_rot = np.array([(M_rot - 1) / 2.0, (N_rot - 1) / 2.0])

    # vectorized inverse‐rotation of all minima positions
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # offsets from rotated‐center
    dr = rows_rot - center_rot[0]
    dc = cols_rot - center_rot[1]

    # inverse rotation back to original coords
    #  [ x_orig ]   [  cos  sin ] [ dc ]   + center_orig[1]
    #  [ y_orig ] = [ -sin  cos ] [ dr ]   + center_orig[0]
    col_orig = cos_t * dc + sin_t * dr + center_orig[1]
    row_orig = -sin_t * dc + cos_t * dr + center_orig[0]

    # round to nearest pixel and filter valid indices
    row_idx = np.round(row_orig).astype(int)
    col_idx = np.round(col_orig).astype(int)
    valid = (
            (row_idx >= 0) & (row_idx < M) &
            (col_idx >= 0) & (col_idx < N)
    )
    return row_idx[valid], col_idx[valid]


def extract_theory_df(I, r_v, theta_v, sigma_small=25, sigma_large=150, angle=60, poly_order=12, spline_lambda=1e-7, e_scale=1.25):
    """
    Process intensity I and coordinate arrays r_v, theta_v.
    Returns a downsampled DataFrame with columns ['pr','E','theta','group_delay','signal'].
    """
    # FFT and initial signal
    fft = np.fft.fft(I, axis=1)
    tot = np.abs(fft[:, 0])
    recovered = skimage.filters.difference_of_gaussians(I, sigma_small, sigma_large, mode='wrap')

    # Rotated minima detection
    args_r = argrelmin_rotated(recovered, angle=angle, axis=1, order=1)
    r_vals, theta_vals = r_v[args_r[0]], theta_v[args_r[1]]

    # Filter by radial range
    mask = (r_vals > 0.13) & (r_vals < 0.27) & (theta_vals > 0) & (theta_vals < 2)
    r_min, theta_min = r_vals[mask], theta_vals[mask]

    # Cluster and select the largest cluster
    labels = DBSCAN(eps=0.1, min_samples=1).fit_predict(np.vstack((r_min * 15, theta_min)).T)
    sel = labels == np.argmax(np.bincount(labels))
    r_min, theta_min = r_min[sel], theta_min[sel]

    # Aggregate by theta
    df_pts = pd.DataFrame({'r': r_min, 'theta': theta_min})
    df_pts = df_pts.groupby('theta', as_index=False)['r'].min()
    r_min, theta_min = df_pts['r'].values, df_pts['theta'].values

    # Contrast weights
    crs = np.max(I, axis=1) - np.min(I, axis=1)
    idxs = np.intersect1d(theta_v, theta_min, return_indices=True)[1]
    cr = crs[idxs]

    # Polynomial smoothing of theta(E)
    E_min = r_min ** 2 / 2
    coeffs = np.polyfit(E_min, theta_min, poly_order, w=cr)
    r_sm = np.linspace(r_min.min(), r_min.max(), len(r_min))
    E = r_sm ** 2 / 2
    theta_sm = np.poly1d(coeffs)(E)
    group_delay = np.gradient(theta_sm, E) * 2

    # Smooth the tot signal
    tot_sm = scipy.interpolate.make_smoothing_spline(r_v ** 2 / 2, tot, lam=spline_lambda)(E)

    theory_df = pd.DataFrame({
        'pr': r_min,
        'E': E * e_scale,
        'theta': theta_sm,
        'group_delay': group_delay,
        'signal': tot_sm / np.max(tot_sm)
    })
    return theory_df


# %%
# Load and filter the theory data
theory_file = r"C:\DATA\4ranges-volume-momspe-2d-sw.dat"
data = []
with open(theory_file) as f:
    for line in f:
        parts = line.strip().split("  ")
        if len(parts) == 6:
            data.append([float(x) for x in parts])
data = np.asarray(data)
data = data[data[:, 0] < 0.35]


def fit_fn(theta, theta0, a, b):
    """Cosine fitting function for angular data."""
    return a * np.cos((theta - theta0) * 2) + b


# Extract data components
pr = data[:, 0]  # Momentum transfer values
theta = data[:, 1] % (2 * np.pi) - np.pi  # Angle values normalized to [-π, π]
intensity = data[:, -1]  # Intensity values

# Reshape to 2D grid based on unique values
grid_size = (len(set(pr)), len(set(theta)))
pr = pr.reshape(grid_size)
theta = theta.reshape(grid_size)
intensity = intensity.reshape(grid_size)
r_values = np.asarray(sorted(np.unique(pr)))
theta_values = np.asarray(sorted(np.unique(theta)))

# Interpolate onto regular grid with 1024 points
RESOLUTION = 1024
grid_indices = (np.linspace(0, len(r_values), RESOLUTION),
                np.linspace(0, len(theta_values), RESOLUTION))
rr, tt = np.meshgrid(*grid_indices)
intensity = scipy.ndimage.map_coordinates(intensity, [rr.flatten(), tt.flatten()])

# Create physical coordinate meshgrid
r_values = np.linspace(min(r_values), max(r_values), RESOLUTION)
theta_values = np.linspace(min(theta_values), max(theta_values), RESOLUTION)
pr, theta = np.meshgrid(r_values, theta_values)
intensity = intensity.reshape(RESOLUTION, RESOLUTION).T

# Apply bump function centered at 0.2 with width 0.1
intensity *= bump((pr.T - 0.2) / 0.1)
I = intensity  # Keep the original variable name for compatibility

# %%
f = px.imshow(I, x=theta_values, y=r_values, origin='lower', aspect='auto').update_layout(
        title="Interlocking Error Bars",
        yaxis_title="Momentum (pr)",
        xaxis_title="Angle (theta)"
)

f2 = go.Figure()

theory_dfs = []

for angle in np.arange(0, 90, 3):
    theory_df = extract_theory_df(I, r_values, theta_values, angle=angle)
    f.add_scatter(x=theory_df['theta'], y=theory_df['pr'], mode='lines', name=f'{angle}°')
    f2.add_scatter(x=theory_df['E'], y=theory_df['group_delay'], mode='lines', name=f'{angle}°')
    theory_dfs.append(theory_df)

for hs in np.arange(100, 200, 4):
    theory_df = extract_theory_df(I, r_values, theta_values, sigma_large=hs)
    f.add_scatter(x=theory_df['theta'], y=theory_df['pr'], mode='lines', name=f'hs={hs}')
    f2.add_scatter(x=theory_df['E'], y=theory_df['group_delay'], mode='lines', name=f'hs={hs}')
    theory_dfs.append(theory_df)

for ls in np.arange(10, 50, 2):
    theory_df = extract_theory_df(I, r_values, theta_values, sigma_small=ls)
    f.add_scatter(x=theory_df['theta'], y=theory_df['pr'], mode='lines', name=f'ls={ls}')
    f2.add_scatter(x=theory_df['E'], y=theory_df['group_delay'], mode='lines', name=f'ls={ls}')
    theory_dfs.append(theory_df)

# Show the figure
f.show()
f2.update_layout(
        title="Group Delay vs Energy (Theory)",
        yaxis_title="Group Delay",
        xaxis_title="Energy (E)",
        yaxis_range=[0, 250],
        xaxis_range=[0.015, 0.035],
).show()

# %%
# Load experimental data
fname_exp = r"J:\ctgroup\Edward\DATA\VMI\20220613\xe005_e_calibrated.h5"
base_data = pd.read_hdf(fname_exp, key="data")

# Filter and symmetrize
data = (base_data
        .query("sqrt(px**2+py**2+pz**2)<0.4")
        .query("abs(py)<0.5")
        .query("pz>0")
        .reset_index(drop=True))
data_sym = data.copy()
data_sym[['px', 'py', 'pz']] *= -1
data = pd.concat([data, data_sym], ignore_index=True)

# Compute spherical coordinates
data['pr'] = np.linalg.norm(data[['px', 'py', 'pz']].values, axis=1)
data['phi'] = np.arctan2(data['px'], data['pz'])
data['theta'] = np.arccos(data['px'] / data['pr'])
data['E'] = data['pr'] ** 2 / 2

# Build 2D histogram
RES = 1024
h2d, re_edges, pe_edges = np.histogram2d(data['pr'], data['phi'], bins=RES)
r_vals = 0.5 * (re_edges[:-1] + re_edges[1:])
theta_vals = 0.5 * (pe_edges[:-1] + pe_edges[1:])
I_exp = (h2d.T * bump((r_vals - 0.25) / 0.1)).T

# %%
# create same plots as theory
f_exp = px.imshow(I_exp, x=theta_vals, y=r_vals, origin='lower', aspect='auto').update_layout(
        title="Experimental Interlocking Error Bars",
        yaxis_title="Momentum (pr)",
        xaxis_title="Angle (phi)"
)
f2_exp = go.Figure()
exp_dfs = []

for angle in np.arange(0, 90, 3):
    df = extract_theory_df(I_exp, r_vals, theta_vals, angle=angle, e_scale=1.0)
    f_exp.add_scatter(x=df['theta'], y=df['pr'], mode='lines', name=f'{angle}°')
    f2_exp.add_scatter(x=df['E'], y=df['group_delay'], mode='lines', name=f'{angle}°')
    exp_dfs.append(df)

for hs in np.arange(120, 180, 4):
    df = extract_theory_df(I_exp, r_vals, theta_vals, sigma_large=hs, e_scale=1.0)
    f_exp.add_scatter(x=df['theta'], y=df['pr'], mode='lines', name=f'hs={hs}')
    f2_exp.add_scatter(x=df['E'], y=df['group_delay'], mode='lines', name=f'hs={hs}')
    exp_dfs.append(df)

for ls in np.arange(10, 40, 2):
    df = extract_theory_df(I_exp, r_vals, theta_vals, sigma_small=ls, e_scale=1.0)
    f_exp.add_scatter(x=df['theta'], y=df['pr'], mode='lines', name=f'ls={ls}')
    f2_exp.add_scatter(x=df['E'], y=df['group_delay'], mode='lines', name=f'ls={ls}')
    exp_dfs.append(df)

# %%
f_exp.show()
f2_exp.update_layout(
        title="Experimental Group Delay vs Energy",
        yaxis_title="Group Delay",
        xaxis_title="Energy (E)",
        yaxis_range=[0, 250],
        xaxis_range=[0.015, 0.035],
).show()

combined_fig = go.Figure().update_layout(
        title="Group Delay vs Energy (Combined Theory and Experiment)",
        yaxis_title="Group Delay",
        xaxis_title="Energy (E)",
        yaxis_range=[0, 200],
        xaxis_range=[0.015, 0.035],
        legend=dict(x=0.01, y=1, traceorder='normal', orientation='h')
)

for i, df in enumerate(theory_dfs):
    combined_fig.add_scatter(x=df['E'], y=df['group_delay'], mode='lines', line_color='red', line_width=1, name='Theory', showlegend=(i == 0))
for i, df in enumerate(exp_dfs):
    combined_fig.add_scatter(x=df['E'], y=df['group_delay'], mode='lines', line_color='blue', line_width=1, name='Experiment', showlegend=(i == 0))

combined_fig.show()

# %%
combined_fig_2 = go.Figure().update_layout(
        title="Phase Delay (Combined Theory and Experiment)",
        yaxis_title="Momentum (pr)",
        xaxis_title="Angle (theta)",
        yaxis_range=[0, 200],
        xaxis_range=[0.015, 0.035],
        legend=dict(x=0.01, y=1, traceorder='normal', orientation='h')
)

for i, df in enumerate(theory_dfs):
    combined_fig_2.add_scatter(x=df['E'], y=2 * df['theta'] / df['E'], mode='lines', line_color='red', line_width=1, name='Theory',
                               showlegend=(i == 0))
for i, df in enumerate(exp_dfs):
    combined_fig_2.add_scatter(x=df['E'], y=2 * df['theta'] - 2.62941 / df['E'], mode='lines', line_color='blue', line_width=1, name='Experiment',
                               showlegend=(i == 0))
combined_fig_2.show()
