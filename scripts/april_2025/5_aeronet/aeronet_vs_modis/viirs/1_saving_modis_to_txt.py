from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
import glob
import datetime
import sys

import os
import calendar

# ---------------------------
# 1. Compute DOY range
# ---------------------------
def doy_range_from_month(year, month):
    first_day = datetime.date(year, month, 1)
    last_day  = datetime.date(year, month, calendar.monthrange(year, month)[1])

    doy_start = first_day.timetuple().tm_yday
    doy_end   = last_day.timetuple().tm_yday

    return doy_start, doy_end


# ---------------------------
# 2. Extract YYYY and DOY from MODIS filename
# ---------------------------
def extract_year_doy(fname):
    """
    MxD04_L2.AYYYYDDD.HHMM...
    """
    base = os.path.basename(fname)
    token = base.split(".")[1]  # 'AYYYYDDD'
    year = int(token[1:5])
    doy  = int(token[5:8])
    return year, doy


# ---------------------------
# 3. Select files in month
# ---------------------------
def select_modis_files_by_month(file_path, year, month):
    doy_start, doy_end = doy_range_from_month(year, month)

    modis_files = sorted(glob.glob(os.path.join(file_path, "AERDB_L2_VIIRS_SNPP.A*.nc")))

    selected = []
    for f in modis_files:
        f_year, f_doy = extract_year_doy(f)

        if f_year == year and doy_start <= f_doy <= doy_end:
            selected.append(f)

    return selected

# Extract variables
file_path = '/scratch/nld6854/earthcare/modis/VIIRS/hdffiles/'
modis_files = select_modis_files_by_month(file_path,2025,11)
print(modis_files[0],modis_files[-1])

def process_viirs_file(fname):
    try:

        # open NetCDF
        ds = xr.open_dataset(fname, mask_and_scale=True)

        # ---- variables ----
        aod = ds["Aerosol_Optical_Thickness_550_Land_Best_Estimate"]
        lat = ds["Latitude"]
        lon = ds["Longitude"]

        # convert to numpy
        aod = aod.astype(np.float32).values
        lat = lat.values
        lon = lon.values

        # ---- physical range filter ----
        aod[(aod <= 0.0005) | (aod >= 5.0)] = np.nan

        # ---- dimensions (VIIRS usually uses Rows/Columns) ----
        dims = aod.shape
        dims = ("y", "x") if len(dims) == 2 else aod.dims

        out = xr.Dataset(
               {"aod": (dims, aod),
                "lat": (dims, lat),
                "lon": (dims, lon),})

        # flatten
        df = (out.stack(point=dims)
               .dropna(dim="point")
               .to_dataframe()
               .reset_index(drop=True))

        # ---- time handling (VIIRS correct way) ----
        if "time_coverage_start" in ds.attrs:
            df["time"] = pd.to_datetime(ds.attrs["time_coverage_start"])
        else:
            df["time"] = pd.NaT

        df = df.dropna(subset=["time"])

        return df

    except Exception as e:
        print(f"Failed: {fname} ({e})")
        return None

results = []

with ProcessPoolExecutor(max_workers=8) as exe:
    for df in exe.map(process_viirs_file, modis_files):
        if df is not None:
            results.append(df)

print('appending finished')
import os

out_dir = file_path+'monthly_aod'
os.makedirs(out_dir, exist_ok=True)

for df in results:
    # Add month column for grouping
    df["month"] = df["time"].dt.strftime("%Y%m")

    # Keep only essential columns
    df = df[["time", "lat", "lon", "aod", "month"]]

    # Loop over months (usually only 1 month per file)
    for month, g in df.groupby("month"):
        out_file = os.path.join(out_dir, f"modis_aod_{month}.csv")

        # Append to file (write header only if file doesn't exist)
        #g.drop(columns="month").to_csv(
        #    out_file,
        #    mode="a",
        #    header=not os.path.exists(out_file),
        #    index=False,
        #    date_format='%Y-%m-%d %H:%M:%S'
        #)

        g = g.copy()
        g["time"] = g["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        g.drop(columns="month").to_csv(
            out_file,
            mode="a",
            header=not os.path.exists(out_file),
            index=False)

print('saving files finished')

# 1. Read as strings
df = pd.read_csv(out_file, dtype={"time": str})

# 2. Parse mixed datetime formats safely
df["time"] = pd.to_datetime(
    df["time"],
    format="mixed",   # <-- KEY FIX
    errors="coerce"
)

# 3. Assert nothing failed
bad = df["time"].isna().sum()
print("NaT rows:", bad)

# 4. Enforce full datetime string
df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

# 5. Rewrite file (overwrite)
df.to_csv(out_file, index=False)

print("Fixed:", out_file)

