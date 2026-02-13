import pandas as pd
import matplotlib.pyplot as plt

def plot_timeseries(df,var1,var):
    fig,ax = plt.subplots(1,figsize=(10,5))
    ax.plot(df["Month_dt"], df[var1], marker='o', label=var1)
    ax.fill_between(df["Month_dt"],df[var1]-df[var1+"_std"],df[var1]+df[var1+"_std"],color='blue',alpha=0.3)
    ax.plot(df["Month_dt"], df[var], marker='s',label=var)
    ax.fill_between(df["Month_dt"],df[var]-df[var+"_std"],df[var]+df[var+"_std"],color='orange',alpha=0.3)
    ax.set_ylabel("AOD",fontsize=15)
    ax.set_title("Monthly AOD Comparison: "+var1+" vs "+var,fontsize=15)
    ax.set_ylim(-0.1,0.7)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(var+'_'+var1+'.jpg')
# Create the dataset
data = {
    "Month": ["12.2024","01.2025","02.2025","03.2025","04.2025","05.2025",
              "06.2025","07.2025","08.2025","09.2025","10.2025","11.2025"],
    "AERONET": [0.13,0.15,0.19,0.2,0.22,0.25,0.3,0.26,0.28,0.24,0.17,0.15],
    "AERONET_std": [0.23,0.22,0.25,0.26,0.29,0.24,0.25,0.21,0.18,0.21,0.17,0.21],
    "ATLID":   [0.18,0.19,0.22,0.23,0.26,0.25,0.3,0.27,0.27,0.23,0.17,0.18],
    "ATLID_std": [0.17,0.18,0.22,0.23,0.22,0.21,0.2,0.17,0.17,0.18,0.16,0.15]
}

df = pd.DataFrame(data)

# Convert Month to datetime for proper plotting
df["Month_dt"] = pd.to_datetime(df["Month"], format="%m.%Y")
plot_timeseries(df,'AERONET','ATLID')

# Dataset
data = {
    "Month": ["12.2024","01.2025","02.2025","03.2025","04.2025","05.2025",
              "06.2025","07.2025","08.2025","09.2025","10.2025","11.2025"],
    "AERONET": [0.12,0.12,0.14,0.15,0.15,0.17,0.2,0.16,0.18,0.14,0.12,0.11],
    "AERONET_std": [0.15,0.14,0.15,0.16,0.16,0.18,0.14,0.14,0.12,0.13,0.13,0.13],
    "Aqua":    [0.12,0.12,0.14,0.15,0.15,0.18,0.22,0.18,0.19,0.14,0.11,0.10],
    "Aqua_std": [0.13,0.14,0.15,0.15,0.15,0.19,0.16,0.14,0.13,0.14,0.12,0.12]
}

df = pd.DataFrame(data)

# Convert Month to datetime
df["Month_dt"] = pd.to_datetime(df["Month"], format="%m.%Y")

plot_timeseries(df,'AERONET','Aqua')

# Create DataFrame
data = {
    "Month": ["12.2024","01.2025","02.2025","03.2025","04.2025","05.2025",
              "06.2025","07.2025","08.2025","09.2025","10.2025","11.2025"],
    "AERONET": [0.11,0.12,0.14,0.15,0.16,0.17,0.2,0.16,0.17,0.14,0.12,0.1],
    "AERONET_std": [0.13,0.14,0.15,0.16,0.17,0.17,0.14,0.13,0.13,0.12,0.13,0.11],
    "Terra":   [0.13,0.13,0.15,0.15,0.15,0.18,0.21,0.18,0.18,0.14,0.12,0.11],
    "Terra_std": [0.15,0.14,0.15,0.17,0.16,0.18,0.15,0.16,0.14,0.11,0.13,0.12]
}

df = pd.DataFrame(data)

# Convert Month to datetime
df['Month_dt'] = pd.to_datetime(df['Month'], format='%m.%Y')

plot_timeseries(df,'AERONET','Terra')


data = {
    "Month": ["12.2024","01.2025","02.2025","03.2025","04.2025","05.2025",
              "06.2025","07.2025","08.2025","09.2025","10.2025","11.2025"],
    "AERONET": [0.11,0.12,0.14,0.15,0.16,0.17,0.2,0.16,0.17,0.14,0.12,0.1],
    "AERONET_std": [0.14,0.15,0.15,0.16,0.18,0.18,0.15,0.15,0.13,0.15,0.13,0.11],
    "VIIRS":   [0.11,0.12,0.12,0.15,0.16,0.18,0.21,0.17,0.18,0.14,0.12,0.1],
    "VIIRS_std": [0.13,0.16,0.16,0.15,0.19,0.29,0.15,0.17,0.13,0.13,0.12,0.11]}

df = pd.DataFrame(data)

# Convert Month to datetime
df['Month_dt'] = pd.to_datetime(df['Month'], format='%m.%Y')

plot_timeseries(df,'AERONET','VIIRS')


data = {
    "Month": ["12.2024","01.2025","02.2025","03.2025","04.2025","05.2025",
              "06.2025","07.2025","08.2025","09.2025","10.2025","11.2025"],
    "Aqua": [0.18,0.21,0.25,0.23,0.25,0.27,0.33,0.29,0.28,0.22,0.14,0.17],
    "Aqua_std": [0.15,0.19,0.21,0.22,0.22,0.25,0.32,0.27,0.28,0.23,0.14,0.16],
    "ATLID": [0.15,0.19,0.21,0.22,0.22,0.25,0.32,0.27,0.28,0.23,0.14,0.16],
    "ATLID_std": [0.15,0.19,0.21,0.22,0.22,0.25,0.32,0.27,0.28,0.23,0.14,0.16],
}

df = pd.DataFrame(data)

# Convert Month to datetime
df['Month_dt'] = pd.to_datetime(df['Month'], format='%m.%Y')

plot_timeseries(df,'Aqua','ATLID')


data = {
    "Month": ["12.2024","01.2025","02.2025","03.2025","04.2025","05.2025",
              "06.2025","07.2025","08.2025","09.2025","10.2025","11.2025"],
    "VIIRS": [0.17,0.17,0.26,0.25,0.31,0.31,0.29,0.34,0.29,0.25,0.22,0.21],
    "VIIRS_std": [0.2,0.17,0.29,0.25,0.38,0.28,0.24,0.35,0.2,0.2,0.24,0.23],
    "ATLID": [0.13,0.16,0.23,0.21,0.25,0.25,0.28,0.27,0.27,0.23,0.19,0.14],
    "ATLID_std": [0.19,0.17,0.23,0.24,0.25,0.22,0.23,0.19,0.17,0.19,0.19,0.15]
}
df = pd.DataFrame(data)

# Convert Month to datetime
df['Month_dt'] = pd.to_datetime(df['Month'], format='%m.%Y')

plot_timeseries(df,'VIIRS','ATLID')
