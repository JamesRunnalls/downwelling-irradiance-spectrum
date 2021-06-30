import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def spectrum(f_dd=1, f_ds=1, theta_sun=30, beta=0.2606, alpha=1.317, H_oz=0.38, WV=2.5, AM=1, RH=60, P=1013.25):
    """Downwelling irradiance spectrum

    Parameters
    -----------
    f_dd : float, optional
        Fraction of direct downwelling irradiance
    f_ds : float, optional
        Fraction of diffuse downwelling irradiance
    theta_sun : float, optional
        Sun zenith angle (deg)
    beta : float, optional
        Turbidity coefficient (0.16 to 0.50)
    alpha : float, optional
        Angström exponent of aerosol scattering (0.2 to 2)
    H_oz : float, optional
        Ozone scale height
    WV : float, optional
        Precipitable water (0 to 5)
    AM : float, optional
        air mass (type 1 (typical of open-ocean aerosols) to 10 (typical of continental aerosols))
    RH : float, optional
        relative humidity (46 to 91)
    P : float, optional
        nonstandard atmospheric pressure

    Variables
    -----------
    E_d : downwelling irradiance spectrum
    E_ds : diffuse downwelling irradiance
    E_dd : direct component of the downwelling irradiance, representing the sun disk in the sky as light source
    E_dsa : aerosol scattering
    E_dsr : Rayleigh scattering
    E_0 : extraterrestrial solar irradiance corrected for earth-sun distance and orbital eccentricity

    T_r : transmittance of the atmosphere after scattering or absorption of Rayleigh scattering
    T_aa : transmittance of the atmosphere after scattering or absorption of aerosol absorption
    T_as : transmittance of the atmosphere after scattering or absorption of aerosol scattering
    T_oz : transmittance of the atmosphere after scattering or absorption of ozone absorption
    T_o : transmittance of the atmosphere after scattering or absorption of oxygen absorption
    T_wv : transmittance of the atmosphere after scattering or absorption of water vapour absorption
    F_a : aerosol forward scattering probability

    M : atmospheric path length
    M_dash : path length corrected for nonstandard atmospheric pressure

    tau_a : aerosol optical thickness
    omega_a : aerosol single scattering albedo
    wavelength_a : reference wavelength (550nm)
    """

    wavelength_a = 550
    df = pd.read_csv("spectra.csv", sep=";")
    E_0 = np.array(df["E_0"])
    a_o = np.array(df["a_o"])
    a_oz = np.array(df["a_oz"])
    a_wv = np.array(df["a_wv"])
    wavelength = np.array(df["wavelength"])

    a = 0.50572
    b = 6.07995
    c = 1.6364
    M = 1/(math.cos(math.radians(theta_sun)) + a * (90 + b - theta_sun)**-c)  # Kasten and Young (1989)
    M_dash = M * P/(1013.25)
    M_oz = 1.0035 / (math.cos(math.radians(theta_sun))**2 + 0.007) ** 0.5
    tau_a = beta * (wavelength/wavelength_a)**-alpha
    omega_a = (-0.0032 * AM + 0.972) * np.exp(0.000306 * RH)

    B_3 = math.log(1 - (-0.1417 * alpha + 0.82))
    B_1 = B_3 * (1.459 + B_3 * (0.1595 + 0.4129 * B_3))
    B_2 = B_3 * (0.0783 + B_3 * (-0.3824 - 0.5874 * B_3))
    F_a = 1 - 0.5 * np.exp(B_1 + B_2 * math.cos(math.radians(theta_sun))) * math.cos(math.radians(theta_sun))

    T_r = np.exp(-M_dash/(115.6404 * (wavelength/1000) ** 4 - 1.335 * (wavelength/1000) ** 2))  # 2.44
    T_aa = np.exp(-(1 - omega_a) * tau_a * M)  # 2.45
    T_as = np.exp(-omega_a * tau_a * M)  # 2.46
    T_oz = np.exp(-a_oz * H_oz * M_oz)  # 2.47
    T_o = np.exp((-1.41 * a_o * M_dash)/(1 + 118.3 * a_o * M_dash)**0.45)  # 2.48
    T_wv = np.exp((-0.2385 * a_wv * WV * M)/(1 + 20.07 * a_wv * WV * M)**0.45)  # 2.49

    E_dd = E_0 * math.cos(math.radians(theta_sun)) * T_r * T_aa * T_as * T_oz * T_o * T_wv  # 2.41
    E_dsr = 0.5 * E_0 * math.cos(math.radians(theta_sun)) * (1 - T_r**0.95) * T_aa * T_oz * T_o * T_wv  # 2.42
    E_dsa = E_0 * math.cos(math.radians(theta_sun)) * T_r**1.5 * T_aa * T_oz * T_o * T_wv * (1 - T_as) * F_a  # 2.43
    E_ds = E_dsr + E_dsa
    E_d = f_dd * E_dd + f_ds * E_ds  # 2.40
    return tau_a, T_r, T_aa, T_as, T_oz, T_o, T_wv, E_dd, E_dsr, E_dsa, E_ds, E_d, wavelength, M_dash


reference = {}
for file in ['twv.FWD', 'edsa.FWD', 'taa.FWD', 'taua.FWD', 'tas.FWD', 'to2.FWD', 'to3.FWD', 'tr.FWD', 'data.FWD', 'edsr.FWD']:
    df = pd.read_csv(file, skiprows=12, header=None, sep="\t")
    df.columns = ["wavelength", "value", "None"]
    reference[file] = df

tau_a, T_r, T_aa, T_as, T_oz, T_o, T_wv, E_dd, E_dsr, E_dsa, E_ds, E_d, wavelength, M_dash = spectrum()

fig, axs = plt.subplots(5, 2)
axs[0, 0].plot(wavelength, tau_a, 'k-')
axs[0, 0].plot(reference["taua.FWD"]["wavelength"], reference["taua.FWD"]["value"], 'r-')
axs[0, 0].set_title('tau_a')
axs[0, 1].plot(wavelength, T_r, 'k-')
axs[0, 1].plot(reference["tr.FWD"]["wavelength"], reference["tr.FWD"]["value"], 'r-')
axs[0, 1].set_title('T_r')
axs[1, 0].plot(wavelength, T_aa, 'k-')
axs[1, 0].plot(reference["taa.FWD"]["wavelength"], reference["taa.FWD"]["value"], 'r-')
axs[1, 0].set_title('T_aa')
axs[1, 1].plot(wavelength, T_as, 'k-')
axs[1, 1].plot(reference["tas.FWD"]["wavelength"], reference["tas.FWD"]["value"], 'r-')
axs[1, 1].set_title('T_as')
axs[2, 0].plot(wavelength, T_oz, 'k-')
axs[2, 0].plot(reference["to3.FWD"]["wavelength"], reference["to3.FWD"]["value"], 'r-')
axs[2, 0].set_title('T_oz')
axs[2, 1].plot(wavelength, T_o, 'k-')
axs[2, 1].plot(reference["to2.FWD"]["wavelength"], reference["to2.FWD"]["value"], 'r-')
axs[2, 1].set_title('T_o')
axs[3, 0].plot(wavelength, T_wv, 'k-')
axs[3, 0].plot(reference["twv.FWD"]["wavelength"], reference["twv.FWD"]["value"], 'r-')
axs[3, 0].set_title('T_wv')
axs[3, 1].plot(wavelength, E_dsr, 'k-')
axs[3, 1].plot(reference["edsr.FWD"]["wavelength"], reference["edsr.FWD"]["value"], 'r-')
axs[3, 1].set_title('E_dsr')
axs[4, 0].plot(wavelength, E_dsa, 'k-')
axs[4, 0].plot(reference["edsa.FWD"]["wavelength"], reference["edsa.FWD"]["value"], 'r-')
axs[4, 0].set_title('E_dsa')
axs[4, 1].plot(wavelength, E_d, 'k-')
axs[4, 1].plot(reference["data.FWD"]["wavelength"], reference["data.FWD"]["value"], 'r-')
axs[4, 1].set_title('E_d')
plt.show()

print(np.mean(np.array(reference["edsa.FWD"]["value"])/E_dsa[50:]))
