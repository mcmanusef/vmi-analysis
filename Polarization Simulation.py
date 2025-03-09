import numpy as np
import matplotlib.pyplot as plt

c_nmfs = 299.792458
# ---------------------------------------------------------------------------------
# Utility Modules
# ---------------------------------------------------------------------------------


def fft_time_to_freq(E_t, t):
    """
    Forward Fourier Transform
    E_t: Time-domain electric field
    t:   Time array
    Returns frequency array (omega), spectral electric field E_w
    """
    dt = t[1] - t[0]
    N = len(t)
    E_w = np.fft.fft(E_t, norm="ortho")
    freq = np.fft.fftfreq(N, d=dt)
    omega = 2.0 * np.pi * freq
    return omega, E_w


def ifft_freq_to_time(E_w):
    """
    Inverse Fourier Transform
    E_w: Frequency-domain electric field
    Returns E_t in time domain
    """
    E_t = np.fft.ifft(E_w, norm="ortho")
    return E_t


def phase_factor(n, L, omega, c_nmfs=299):
    """
    Phase shift factor for dispersion in spectral domain
    n: Refractive index (can be array if wavelength-dependent)
    L: Optical path length (meters)
    omega: angular frequencies array
    c: speed of light in m/s
    Returns array of phase factors exp(i phi)
    """
    phi = (n * omega * L * 1e9) / c_nmfs
    return np.exp(1j * phi)


# ---------------------------------------------------------------------------------
# Polarization / Jones Calculus
# ---------------------------------------------------------------------------------


def jones_matrix_waveplate(retardance, orientation):
    """
    Returns the Jones matrix for a wave plate of given retardance (phase difference)
    and orientation (angle in radians).
    For orientation theta, the Jones matrix is:
       R(-theta) * diag(1, e^{i*retardance}) * R(theta)
    where R(theta) is rotation matrix.
    """
    c = np.cos(orientation)
    s = np.sin(orientation)
    j_rot_minus = np.array([[c, s], [-s, c]], dtype=np.complex128)
    j_rot_plus = np.array([[c, -s], [s, c]], dtype=np.complex128)
    diag = np.array([[1.0, 0.0], [0.0, np.exp(1j * retardance)]], dtype=np.complex128)
    return j_rot_minus @ diag @ j_rot_plus


def apply_jones_matrix(E_jones, J):
    """
    Applies the Jones matrix J to the input Jones vector E_jones.
    E_jones shape: (2, num_frequencies)
    J shape: (2,2) or (2,2,num_frequencies) if wavelength dependent
    Returns the new Jones vector.
    """
    if len(J.shape) == 2:
        return J @ E_jones
    else:
        # J is frequency-dependent
        out = np.zeros_like(E_jones, dtype=np.complex128)
        for i in range(E_jones.shape[1]):
            out[:, i] = J[:, :, i] @ E_jones[:, i]
        return out


# ---------------------------------------------------------------------------------
# Dispersion Module
# ---------------------------------------------------------------------------------


def material_refractive_index(wavelength_nm, material="BK7"):
    """
    Placeholder function for refractive index.
    Extend or replace with real data or Sellmeier equations as needed.
    wavelength_nm: scalar or array of wavelength in nm
    material: string specifying the material
    Returns n: refractive index (same shape as wavelength_nm)
    """
    # Simple placeholders for demonstration.
    # Insert real dispersion data or Sellmeier-based calculations here.
    # Example: BK7 approximate formula for the visible range
    # (This is not accurate for a broad range, just for demonstration).
    # n^2 - 1 = (1.040e-2 * λ^2) / (λ^2 - 6.006e-3) + ...
    # Using constants would be best in a real code.
    # For now, just a constant index for demonstration:
    wavelength_nm = np.clip(wavelength_nm, 400, 1500)
    if material == "BK7":
        n = (
            lambda λ: (
                1
                + (1.03961212 * λ**2) / (λ**2 - 0.00600069867)
                + (0.231792344 * λ**2) / (λ**2 - 0.0200179144)
                + (1.01046945 * λ**2) / (λ**2 - 103.560653)
            )
            ** 0.5
        )
        return n(wavelength_nm / 1000)
    elif material == "Fused Silica":
        n = (
            lambda λ: (
                1
                + (0.6961663 * λ**2) / (λ**2 - 0.0684043**2)
                + (0.4079426 * λ**2) / (λ**2 - 0.1162414**2)
                + (0.8974794 * λ**2) / (λ**2 - 9.896161**2)
            )
            ** 0.5
        )
        return n(wavelength_nm / 1000)
    elif material == "Calcite (O)":
        n = (
            lambda λ: (
                1
                + 0.73358749
                + (0.96464345 * λ**2) / (λ**2 - 1.9435203e-2)
                + (1.82831454 * λ**2) / (λ**2 - 120)
            )
            ** 0.5
        )
        return n(wavelength_nm / 1000)
    elif material == "Calcite (E)":
        n = (
            lambda λ: (
                1
                + 0.35859695
                + (0.82427830 * λ**2) / (λ**2 - 1.06689543e-2)
                + (0.14429128 * λ**2) / (λ**2 - 120)
            )
            ** 0.5
        )
        return n(wavelength_nm / 1000)
    else:
        return 1.0 + 0.0 * wavelength_nm


def apply_dispersion(E_jones, omega, wavelength_nm, thickness_m, material="BK7"):
    """
    Applies the material dispersion as a phase shift in the spectral domain.
    E_jones: (2, num_frequencies) Jones vector in freq domain
    omega:   array of angular frequencies
    wavelength_nm: array of wavelengths for each frequency component
    thickness_m: thickness in meters
    material: which material
    Returns updated Jones vector.
    """
    n_vals = material_refractive_index(wavelength_nm, material=material)
    # Phase shift for each frequency
    ph = phase_factor(n_vals, thickness_m, omega)
    # Apply identical phase shift to both polarization components
    E_jones[0, :] *= ph
    E_jones[1, :] *= ph
    return E_jones


# ---------------------------------------------------------------------------------
# Optical Element Modules
# ---------------------------------------------------------------------------------


def gvd_compensation_plate(
    E_jones,
    omega,
    wavelength_nm,
    delay_fs=0,
    long_wavelength_nm=1030.0,
    short_wavelength_nm=515.0,
    dispersive=False,
):
    if dispersive:
        opl_diff = delay_fs * c_nmfs  # optical path length difference
        # Long sees ordinary, short sees extraordinary
        n_long = material_refractive_index(long_wavelength_nm, material="Calcite (O)")
        n_short = material_refractive_index(short_wavelength_nm, material="Calcite (E)")

        thickness_nm = opl_diff / (n_long - n_short)
        thickness_m = thickness_nm * 1e-9
        # print(f"Thickness: {thickness_m*1e3:.3f} mm")
        n_o = material_refractive_index(wavelength_nm, material="Calcite (O)")
        n_e = material_refractive_index(wavelength_nm, material="Calcite (E)")
        pho = phase_factor(n_o, thickness_m, omega)
        phe = phase_factor(n_e, thickness_m, omega)
        E_jones[0, :] *= pho
        E_jones[1, :] *= phe
        return E_jones

    else:
        E_jones[1, :] *= np.exp(-1j * delay_fs * omega)
        return E_jones


def super_achromatic_quarter_waveplate(E_jones, wavelength_nm, orientation_deg=45.0):
    """
    Super-achromatic quarter wave plate oriented at 45 degrees.
    We'll assume perfect quarter wave across the entire bandwidth for simplicity.
    Real designs would have wavelength-dependent retardance.
    """
    orientation = np.deg2rad(orientation_deg)
    # Quarter wave => retardance = pi/2
    # For broad range, one might have a function retardance(λ). We'll assume constant.
    retardance = np.pi / 2
    J = jones_matrix_waveplate(retardance, orientation)
    E_jones_out = apply_jones_matrix(E_jones, J)
    return E_jones_out


def zero_order_half_waveplate(
    E_jones, wavelength_nm, orientation_deg=0.0, center_wavelength_nm=1030.0
):
    """
    Zero-order half wave plate centered at 1030 nm.
    We assume it acts as a half wave plate at 1030 nm exactly.
    For 515 nm, the retardance might differ. For simplicity, assume it doesn't match 515 nm.
    We'll show an example:
       If wavelength is 1030 nm +/- small, retardance ~ pi
       If wavelength is significantly different, it might vary.
    For demonstration, let's do a linear interpolation around 1030 nm.
    """
    orientation = np.deg2rad(orientation_deg)
    # We'll define a simple approach:
    # At λ=1030 nm => retardance = pi
    # At λ=515 nm  => retardance = something else (we'll assume 2*pi? Just for demonstration)
    # You can define your own function. We'll do a naive linear interpolation.
    retardance_arr = np.zeros_like(wavelength_nm, dtype=np.float64)
    for i, wl in enumerate(wavelength_nm):
        if wl < 515:
            # Should not happen in typical use, but let's just clamp
            wl = 515
        if wl > 1030:
            # clamp
            wl = 1030

        # Simple linear interpolation between 515 nm -> 2*pi and 1030 nm -> pi
        # This is purely illustrative. Replace with actual data for real wave plates.
        slope = (np.pi - 2.0 * np.pi) / (1030.0 - 515.0)
        intercept = 2.0 * np.pi - slope * 515.0
        retardance_arr[i] = slope * wl + intercept

    # Build a frequency-dependent Jones matrix
    J_all = np.zeros((2, 2, len(wavelength_nm)), dtype=np.complex128)
    for i in range(len(wavelength_nm)):
        J_all[:, :, i] = jones_matrix_waveplate(retardance_arr[i], orientation)

    E_jones_out = apply_jones_matrix(E_jones, J_all)
    return E_jones_out


def glass_window(E_jones, omega, wavelength_nm, thickness_m, material="BK7"):
    """
    Glass window applying dispersion.
    """
    return apply_dispersion(
        E_jones, omega, wavelength_nm, thickness_m, material=material
    )


# ---------------------------------------------------------------------------------
# Core Simulation Classes
# ---------------------------------------------------------------------------------


class TwoColorPulse:
    """
    Represents the initial two-color pulse in the time domain.
    We define amplitude, central wavelength, and pulse duration (intensity FWHM).
    We assume perpendicular polarization: E_x for 1030 nm, E_y for 515 nm.
    """

    def __init__(
        self,
        amp_long=1.0,
        amp_short=1.0,
        wl_long_nm=1030.0,
        wl_short_nm=515.0,
        duration_fs=130.0,
        time_window_fs=2000.0,
        num_points=2**14,
        normalize=True,
    ):
        self.amp_long = amp_long
        self.amp_short = amp_short
        if normalize:
            self.amp_long, self.amp_short = (
                self.amp_long / np.sqrt(self.amp_long**2 + self.amp_short**2),
                self.amp_short / np.sqrt(self.amp_long**2 + self.amp_short**2),
            )
        self.wl_long_nm = wl_long_nm
        self.wl_short_nm = wl_short_nm
        self.duration_fs = duration_fs
        self.time_window_fs = time_window_fs
        self.num_points = num_points

        # Prepare time array
        self.t = np.linspace(
            -0.5 * self.time_window_fs, 0.5 * self.time_window_fs, self.num_points
        )

        # Create the time-domain fields
        # We model them as Gaussian pulses in time
        self.Et_x, self.Et_y = self._create_time_domain_pulses()

    def _create_time_domain_pulses(self):
        # Convert duration_fs (intensity FWHM) to standard deviation
        # Intensity FWHM of Gaussian -> E-field FWHM is shorter by factor ~ 1.4142
        # so we'll approximate field_fwhm ~ duration_fs / np.sqrt(2)
        field_fwhm_fs = self.duration_fs / np.sqrt(2)
        sigma_t = field_fwhm_fs / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Angular frequency for each central wavelength
        omega_1030 = 2.0 * np.pi * c_nmfs / self.wl_long_nm
        omega_515 = 2.0 * np.pi * c_nmfs / self.wl_short_nm

        E_x = (
            self.amp_long
            * np.exp(-0.5 * (self.t**2) / (sigma_t**2))
            * np.exp(1j * omega_1030 * self.t)
        )
        E_y = (
            self.amp_short
            * np.exp(-0.5 * (self.t**2) / (sigma_t**2))
            * np.exp(1j * omega_515 * self.t)
        )
        return E_x, E_y

    def get_time_domain_fields(self):
        return self.t, self.Et_x, self.Et_y


class Simulation:
    """
    Main Controller to orchestrate the simulation flow.
    """

    def __init__(self, pulse: TwoColorPulse):
        self.pulse = pulse

        # Convert time-domain fields to frequency domain
        self.omega, self.Ew = self._build_jones_in_freq_domain()
        # We also track a wavelength array
        # lambda = 2 pi c / omega, but we must be mindful of negative frequencies from FFT
        # We'll reorder arrays properly after the FFT
        self.wavelength_nm = self._make_wavelength_array(self.omega)

    def _build_jones_in_freq_domain(self):
        t, Ex_t, Ey_t = self.pulse.get_time_domain_fields()
        omega, Ex_w = fft_time_to_freq(Ex_t, t)
        _, Ey_w = fft_time_to_freq(Ey_t, t)

        omega_sorted = omega
        E_jones = np.vstack((Ex_w, Ey_w))

        return omega_sorted, E_jones

    def _make_wavelength_array(self, omega):
        # For each omega, compute lambda in nm = 2 pi c_nmfs / omega
        # For negative frequencies, we can do something or just keep it. Usually we just mirror.
        # We'll replace negative frequency with a large number for wavelength or handle carefully.
        wl = np.zeros_like(omega)
        for i, w in enumerate(omega):
            if w > 0.0:
                wl[i] = 2.0 * np.pi * c_nmfs / w
            else:
                # Just set a placeholder or negative
                wl[i] = -1.0
        return wl

    def apply_optical_element_gvd_plate(
        self,
        delay_fs=0,
        long_wavelength_nm=1030.0,
        short_wavelength_nm=515.0,
        dispersive=False,
    ):
        self.Ew = gvd_compensation_plate(
            self.Ew,
            self.omega,
            self.wavelength_nm,
            delay_fs=delay_fs,
            long_wavelength_nm=long_wavelength_nm,
            short_wavelength_nm=short_wavelength_nm,
            dispersive=dispersive,
        )

    def apply_optical_element_quarter_wave(self, orientation_deg=45.0):
        self.Ew = super_achromatic_quarter_waveplate(
            self.Ew, self.wavelength_nm, orientation_deg=orientation_deg
        )

    def apply_optical_element_half_wave(self, orientation_deg=0.0):
        self.Ew = zero_order_half_waveplate(
            self.Ew, self.wavelength_nm, orientation_deg=orientation_deg
        )

    def apply_optical_element_glass_window(self, thickness_m=0.001, material="BK7"):
        self.Ew = glass_window(
            self.Ew, self.omega, self.wavelength_nm, thickness_m, material
        )

    def get_time_domain_fields(self):
        """
        Returns the time array and the inverse Fourier transform of the Jones vector
        after all elements have been applied.
        """
        # Rebuild the unsorted version by reversing the sorting used in initialization if needed
        # But simpler: we can just do inverse FFT in the sorted domain if we handle it carefully.
        # We'll reorder back to the original frequency order for ifft.
        # The original indices were from self._build_jones_in_freq_domain. Let's do it carefully.

        # We know that we sorted by frequency. We can invert that sort.
        # But for practical simplicity, let's just do an ifft on the sorted domain as is.

        Ex_w = self.Ew[0, :]
        Ey_w = self.Ew[1, :]

        Ex_t = ifft_freq_to_time(Ex_w)
        Ey_t = ifft_freq_to_time(Ey_w)

        # The time array in the simulation is the same as the pulse's time array
        t = self.pulse.t
        return t, Ex_t, Ey_t

    def get_wavelength_domain_fields(self):
        """
        Return the wavelength array and the current Jones vector in freq domain
        for positive frequencies only. Negative frequencies are typically mirror.
        """
        # We'll only keep positive frequency region for a typical spectral plot
        mask_positive = self.omega > 0
        wl_pos = self.wavelength_nm[mask_positive]
        Ex_w = self.Ew[0, mask_positive]
        Ey_w = self.Ew[1, mask_positive]
        return wl_pos, Ex_w, Ey_w


def plot_stokes_parameters(t, Ex_t, Ey_t, wvln=1030):
    """
    Calculate and plot the Stokes parameters as a function of time.

    Parameters:
    - t: Time array (fs or any consistent unit)
    - Ex_t: Complex electric field component along the x-axis (time domain)
    - Ey_t: Complex electric field component along the y-axis (time domain)
    """
    # Calculate the intensities and cross terms
    I = np.abs(Ex_t) ** 2 + np.abs(Ey_t) ** 2  # Total intensity
    Q = np.abs(Ex_t) ** 2 - np.abs(Ey_t) ** 2  # Difference in intensity
    U = 2 * np.real(Ex_t * np.conj(Ey_t))  # Real part of the cross term
    V = 2 * np.imag(
        Ex_t * np.conj(Ey_t)
    )  # Imaginary part of the cross term (circular polarization)

    # Calculate Cycle-averaged Stokes parameters, which are time-averaged over one cycle
    period = wvln / c_nmfs
    print(period)
    cycle_points = int(period / (t[1] - t[0]))
    print(cycle_points)
    kernel = np.ones(cycle_points) / cycle_points
    I = np.convolve(I, kernel, mode="same")
    Q = np.convolve(Q, kernel, mode="same")
    U = np.convolve(U, kernel, mode="same")
    V = np.convolve(V, kernel, mode="same")

    # Plot Stokes parameters
    plt.figure(figsize=(10, 8))
    plt.suptitle("Stokes Parameters (Cycle Averaged) vs Time")
    plt.subplot(4, 1, 1)
    plt.plot(t, I, label="S0 (Intensity)")
    plt.ylabel("S0")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(t, Q, label="S1 (Linear Polarization: Horizontal/Vertical)")
    plt.ylabel("S1")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t, U, label="S2 (Linear Polarization: Diagonal/Anti-diagonal)")
    plt.ylabel("S2")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(t, V, label="S3 (Circular Polarization)")
    plt.xlabel("Time (fs)")
    plt.ylabel("S3")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


# ---------------------------------------------------------------------------------
# Example of usage
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    # Create initial pulse
    pulse = TwoColorPulse(
        amp_long=1,  # amplitude of 1030 nm
        amp_short=1,  # amplitude of 515 nm
        wl_long_nm=1030.0,
        wl_short_nm=515,
        duration_fs=130.0,  # 130 fs
        time_window_fs=2000.0,
        num_points=2**20,
    )

    # Initialize simulation
    sim = Simulation(pulse)

    # Apply a GVD plate
    sim.apply_optical_element_gvd_plate(delay_fs=170, dispersive=True)
    sim.apply_optical_element_gvd_plate(delay_fs=170, dispersive=True)

    # Apply a super-achromatic quarter wave plate at 45 degrees
    sim.apply_optical_element_quarter_wave(orientation_deg=45)

    # Apply a zero-order half-wave plate at 0 degrees
    sim.apply_optical_element_half_wave(orientation_deg=0.0)

    # sim.apply_optical_element_glass_window(thickness_m=.038, material='BK7')

    # Apply a glass window of thickness 2 mm
    sim.apply_optical_element_glass_window(thickness_m=0.004, material="Fused Silica")

    # Retrieve final time-domain fields
    t, Ex_t_final, Ey_t_final = sim.get_time_domain_fields()
    with open("counter.txt", mode="w") as f:
        for i in range(len(t)):
            f.write(
                f"{t[i]},{np.real(Ex_t_final[i])},{np.imag(Ex_t_final[i])},{np.real(Ey_t_final[i])},{np.imag(Ey_t_final[i])}\n"
            )

    # Retrieve final wavelength-domain fields (positive freq region)
    wl, Ex_w_final, Ey_w_final = sim.get_wavelength_domain_fields()

    # Plot time-domain results
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.suptitle("Time-domain Electric Field")
    ax[0].plot(t, np.real(Ex_t_final), label="Real(Ex)")
    ax[0].plot(t, np.imag(Ex_t_final), label="Imag(Ex)", ls="--")
    tcx = t[np.argmax(np.abs(Ex_t_final))]
    ax[0].axvline(tcx, color="b", ls="--", lw=0.5)

    ax[0].set_ylabel("Ex(t)")
    ax[0].legend()

    ax[1].plot(t, np.real(Ey_t_final), label="Real(Ey)")
    ax[1].plot(t, np.imag(Ey_t_final), label="Imag(Ey)", ls="--")
    ax[1].set_ylabel("Ey(t)")
    ax[1].set_xlabel("Time (fs)")
    tcy = t[np.argmax(np.abs(Ey_t_final))]
    ax[1].axvline(tcy, color="orange", ls="--", lw=0.5)
    ax[1].legend()

    print(t[np.argmax(np.abs(Ex_t_final))], t[np.argmax(np.abs(Ey_t_final))])
    print(t[np.argmax(np.abs(Ex_t_final))] - t[np.argmax(np.abs(Ey_t_final))])

    plt.tight_layout()

    # Plot spectral-domain results
    plt.figure(figsize=(8, 4))
    plt.title("Spectrum and Chirp")
    omega = 2 * np.pi * c_nmfs / wl
    plt.plot(wl, np.abs(Ex_w_final), label="|Ex(λ)|")
    plt.plot(wl, np.abs(Ey_w_final), label="|Ey(λ)|")
    plt.xlim(400, 1200)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral amplitude")
    plt.legend()
    plt.twinx()

    x_nonzero = np.argwhere(np.abs(Ex_w_final) > 0.001)
    y_nonzero = np.argwhere(np.abs(Ey_w_final) > 0.001)
    chirp_x = np.gradient(np.gradient(np.unwrap(np.angle(Ex_w_final)), omega), omega)
    chirp_y = np.gradient(np.gradient(np.unwrap(np.angle(Ey_w_final)), omega), omega)
    plt.scatter(wl[x_nonzero], chirp_x[x_nonzero], label="Phase(Ex)", s=1)
    plt.scatter(wl[y_nonzero], chirp_y[y_nonzero], label="Phase(Ey)", s=1)
    plt.ylabel("Chirp (fs^2)")
    plt.ylim(0, 10000)
    plt.tight_layout()

    plt.figure(figsize=(8, 8)).add_subplot(projection="3d")
    plt.title("Electric Field")
    plt.plot(t, np.real(Ex_t_final), np.real(Ey_t_final))
    plt.xlim(tcx - 100, tcx + 100)
    plt.ylim(-1, 1)
    plt.gca().set_zlim(-1, 1)
    plt.xlabel("Time (fs)")
    plt.ylabel("Ex(t)")
    plt.gca().set_zlabel("Ey(t)")
    plt.tight_layout()

    plt.figure(figsize=(8, 8))
    plt.title("Polarization Path")
    plt.plot(np.real(Ex_t_final), np.real(Ey_t_final), lw=1)
    plt.xlabel("Ex(t)")
    plt.ylabel("Ey(t)")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1, projection="polar")
    plt.title("Polarization Path")
    l = plt.plot(
        np.arctan2(np.real(Ey_t_final), np.real(Ex_t_final)),
        np.sqrt(np.real(Ex_t_final) ** 2 + np.real(Ey_t_final) ** 2),
        lw=1,
    )

    # Plot Stokes parameters
    plot_stokes_parameters(t, Ex_t_final, Ey_t_final)

    plt.show()
