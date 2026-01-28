
import pandas as pd

def load_data_from_xlsx(xlsx_file):
    trip = pd.read_excel(xlsx_file, sheet_name='Trip Data')
    ridermap = pd.read_excel(xlsx_file, sheet_name="Checked in User ID's")
    demo = pd.read_excel(xlsx_file, sheet_name='Customer Demographics')

    trip['Trip Date and Time'] = pd.to_datetime(trip['Trip Date and Time'])
    trip['date'] = trip['Trip Date and Time'].dt.date
    trip['dow'] = trip['Trip Date and Time'].dt.day_name()
    trip['hour'] = trip['Trip Date and Time'].dt.hour

    return trip, ridermap, demo

def any_passenger_age_cohort(trip, ridermap, demo, low, high):
    merged = ridermap.merge(demo, on='User ID', how='left')
    merged['is_band'] = merged['Age'].between(low, high, inclusive='both')
    trip_ids = merged[merged['is_band']].drop_duplicates('Trip ID')['Trip ID']
    return trip[trip['Trip ID'].isin(trip_ids)].copy()

def last_n_days(trip, n=30):
    end = trip['Trip Date and Time'].max()
    start = end - pd.Timedelta(days=n)
    return trip[(trip['Trip Date and Time']>=start) & (trip['Trip Date and Time']<=end)].copy()

def downtown_box_mask(df):
    return (df['Drop Off Latitude'].between(30.262, 30.278)) & (df['Drop Off Longitude'].between(-97.752, -97.732))
