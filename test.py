import os
import pandas as pd
import matplotlib.pyplot as plt
from downwellingirradiancespectrum import spectrum

folder = "60"
theta_sun = 60

reference = {}
for file in ['twv.FWD', 'edsa.FWD', 'taa.FWD', 'taua.FWD', 'tas.FWD', 'to2.FWD', 'to3.FWD', 'tr.FWD', 'data.FWD', 'edsr.FWD']:
    if os.path.isfile(os.path.join(folder, file)):
        df = pd.read_csv(os.path.join(folder, file), skiprows=12, header=None, sep="\t")
        df.columns = ["wavelength", "value", "None"]
        reference[file] = df

wavelength, E_d, E_dd, E_ds, E_dsr, E_dsa, T_r, T_aa, T_as, T_oz, T_o, T_wv, tau_a, M_dash = spectrum(theta_sun=theta_sun, all_parameters=True)

fig, axs = plt.subplots(5, 2)
axs[0, 0].plot(wavelength, tau_a, 'k-')
if "taua.FWD" in reference:
    axs[0, 0].plot(reference["taua.FWD"]["wavelength"], reference["taua.FWD"]["value"], 'r-')
axs[0, 0].set_title('tau_a')
axs[0, 1].plot(wavelength, T_r, 'k-')
if "tr.FWD" in reference:
    axs[0, 1].plot(reference["tr.FWD"]["wavelength"], reference["tr.FWD"]["value"], 'r-')
axs[0, 1].set_title('T_r')
axs[1, 0].plot(wavelength, T_aa, 'k-')
if "taa.FWD" in reference:
    axs[1, 0].plot(reference["taa.FWD"]["wavelength"], reference["taa.FWD"]["value"], 'r-')
axs[1, 0].set_title('T_aa')
axs[1, 1].plot(wavelength, T_as, 'k-')
if "tas.FWD" in reference:
    axs[1, 1].plot(reference["tas.FWD"]["wavelength"], reference["tas.FWD"]["value"], 'r-')
axs[1, 1].set_title('T_as')
axs[2, 0].plot(wavelength, T_oz, 'k-')
if "to3.FWD" in reference:
    axs[2, 0].plot(reference["to3.FWD"]["wavelength"], reference["to3.FWD"]["value"], 'r-')
axs[2, 0].set_title('T_oz')
axs[2, 1].plot(wavelength, T_o, 'k-')
if "to2.FWD" in reference:
    axs[2, 1].plot(reference["to2.FWD"]["wavelength"], reference["to2.FWD"]["value"], 'r-')
axs[2, 1].set_title('T_o')
axs[3, 0].plot(wavelength, T_wv, 'k-')
if "twv.FWD" in reference:
    axs[3, 0].plot(reference["twv.FWD"]["wavelength"], reference["twv.FWD"]["value"], 'r-')
axs[3, 0].set_title('T_wv')
axs[3, 1].plot(wavelength, E_dsr, 'k-')
if "edsr.FWD" in reference:
    axs[3, 1].plot(reference["edsr.FWD"]["wavelength"], reference["edsr.FWD"]["value"], 'r-')
axs[3, 1].set_title('E_dsr')
axs[4, 0].plot(wavelength, E_dsa, 'k-')
if "edsa.FWD" in reference:
    axs[4, 0].plot(reference["edsa.FWD"]["wavelength"], reference["edsa.FWD"]["value"], 'r-')
axs[4, 0].set_title('E_dsa')
axs[4, 1].plot(wavelength, E_d, 'k-')
if "data.FWD" in reference:
    axs[4, 1].plot(reference["data.FWD"]["wavelength"], reference["data.FWD"]["value"], 'r-')
axs[4, 1].set_title('E_d')
plt.show()