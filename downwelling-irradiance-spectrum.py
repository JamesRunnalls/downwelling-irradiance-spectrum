import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def spectrum(f_dd=1, f_ds=1, theta_sun=30, beta=0.2606, alpha=1.317, H_oz=0.38, WV=2.5, AM=10, RH=0.69, P=1013.25):
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
        relative humidity (0.46 to 0.91)
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

    M = 1/(math.cos(math.radians(theta_sun)) + 0.50572 * (90 + 6.07995 - theta_sun)**-1.6364)  # Kasten and Young (1989)
    M_dash = M * P/(1013.25)
    M_oz = 1.0035 / (math.cos(math.radians(theta_sun))**2 + 0.007) ** 0.5
    tau_a = beta * (wavelength/wavelength_a)**-alpha
    omega_a = (-0.0032 * AM + 0.972) * np.exp(3.06*10**-4 * RH)

    B_3 = math.log(1 - (-0.1417 * alpha + 0.82))
    B_1 = B_3 * (1.459 + B_3 * (0.1595 + 0.4129 * B_3))
    B_2 = B_3 * (0.0783 + B_3 * (-0.3824 - 0.5874 * B_3))
    F_a = 1 - 0.5 * np.exp(B_1 + B_2 * math.cos(math.radians(theta_sun))) * math.cos(math.radians(theta_sun))

    T_r = np.exp(-M_dash/(115.6404 * wavelength ** 4 - 1.335 * wavelength ** 2))  # 2.44
    T_aa = np.exp(-(1 - omega_a) * tau_a * M)  # 2.45
    T_as = np.exp(-omega_a * tau_a * M)  # 2.46
    T_oz = np.exp(-a_oz * H_oz * M_oz)  # 2.47
    T_o = np.exp((-1.41 * a_o * M_dash)/(1 + 118.3 * a_o * M_dash)**0.45)  # 2.48
    T_wv = np.exp((-0.2385 * a_wv * WV * M)/(1 + 20.07 * a_wv * WV * M)**0.45)  # 2.49

    E_dd = E_0 * math.cos(math.radians(theta_sun)) * T_r * T_aa * T_as * T_oz * T_o * T_wv  # 2.41
    E_dsr = 0.5 * E_0 * math.cos(math.radians(theta_sun)) * (1 - T_r**0.95) * T_aa * T_oz * T_o * T_wv  # 2.42
    E_dsa = E_0 * math.cos(math.radians(theta_sun)) * T_r**1.5 * T_aa * T_oz * T_wv * (1 - T_as) * F_a  # 2.43
    E_ds = E_dsr + E_dsa
    E_d = f_dd * E_dd + f_ds * E_ds  # 2.40
    return E_d, wavelength


E_d, wavelength = spectrum()
plt.plot(wavelength, E_d)
plt.show()
