import os
import pandas as pd
import numpy as np
import psycopg2
import gspread
from prophet import Prophet
from datetime import datetime, timedelta
from google.ads.googleads.client import GoogleAdsClient
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# =========================================
# CONFIG
# =========================================

DEVELOPER_TOKEN = os.environ["DEVELOPER_TOKEN"]
CLIENT_ID       = os.environ["CLIENT_ID"]
CLIENT_SECRET   = os.environ["CLIENT_SECRET"]

MCC_IDS = [
    "7141208780",
    "7309803413",
    "5419872903",
    "8567995305"
]

SHEET_ID     = os.environ["SHEET_ID"]
RAW_TAB      = "Raw_Data"
FORECAST_TAB = "Forecast"

REDSHIFT_HOST = os.environ["REDSHIFT_HOST"]
REDSHIFT_PORT = 5439
REDSHIFT_DB   = os.environ["REDSHIFT_DB"]
REDSHIFT_USER = os.environ["REDSHIFT_USER"]
REDSHIFT_PASS = os.environ["REDSHIFT_PASS"]

# =========================================
# GOOGLE AUTH
# =========================================

SCOPES = [
    "https://www.googleapis.com/auth/adwords",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_credentials():
    TOKEN_PATH = "token.json"
    if not os.path.exists(TOKEN_PATH):
        raise FileNotFoundError("token.json not found.")
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        print("✅ Token refreshed.")
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return creds

creds = get_credentials()
gc    = gspread.authorize(creds)

# =========================================
# GOOGLE ADS CLIENT
# =========================================

def get_ads_client(mcc):
    config = {
        "developer_token":   DEVELOPER_TOKEN,
        "client_id":         CLIENT_ID,
        "client_secret":     CLIENT_SECRET,
        "refresh_token":     creds.refresh_token,
        "login_customer_id": mcc,
        "use_proto_plus":    True,
        "api_version":       "v14"
    }
    return GoogleAdsClient.load_from_dict(config)

# =========================================
# REDSHIFT
# =========================================

def get_adtopia_mapping():
    conn = psycopg2.connect(
        host=REDSHIFT_HOST,
        port=REDSHIFT_PORT,
        dbname=REDSHIFT_DB,
        user=REDSHIFT_USER,
        password=REDSHIFT_PASS
    )
    query = """
    SELECT
        id            AS ad_campaign_id,
        campaign_name AS adtopia_campaign_name
    FROM adtopia2.adtopia_campaigns
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["ad_campaign_id"] = df["ad_campaign_id"].astype(str)
    print("✅ Redshift mapping loaded:", len(df), "rows")
    return df

# =========================================
# FETCH ADS DATA
# =========================================

def fetch_ads_data(mcc):
    client  = get_ads_client(mcc)
    service = client.get_service("GoogleAdsService")

    query_children = """
    SELECT
        customer_client.id,
        customer_client.descriptive_name,
        customer_client.manager
    FROM customer_client
    WHERE customer_client.level = 1
    """

    response = service.search(customer_id=mcc, query=query_children)
    accounts = []
    for row in response:
        if not row.customer_client.manager:
            accounts.append((
                str(row.customer_client.id),
                row.customer_client.descriptive_name
            ))

    print(f"  Accounts found under {mcc}:", len(accounts))

    start_date = (datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date   = datetime.today().strftime('%Y-%m-%d')

    query = f"""
    SELECT
        campaign.id,
        campaign.name,
        segments.date,
        metrics.clicks,
        metrics.impressions,
        metrics.cost_micros,
        metrics.conversions,
        metrics.all_conversions_value
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    """

    rows = []
    for cust_id, name in accounts:
        try:
            resp = service.search(customer_id=cust_id, query=query)
            for r in resp:
                cost = r.metrics.cost_micros / 1_000_000
                rows.append({
                    "MCC_ID":      mcc,
                    "Account_ID":  cust_id,
                    "Account":     name,
                    "Campaign_ID": str(r.campaign.id),
                    "Campaign":    r.campaign.name,
                    "Date":        r.segments.date,
                    "Clicks":      r.metrics.clicks,
                    "Impressions": r.metrics.impressions,
                    "Cost":        round(cost, 2),
                    "Conversions": round(r.metrics.conversions, 2),
                    "ROAS":        round(r.metrics.all_conversions_value / cost, 2) if cost > 0 else 0
                })
        except Exception as e:
            print(f"  ⚠️ Skipping account {cust_id}: {e}")

    return pd.DataFrame(rows)

# =========================================
# WRITE TO SHEET
# =========================================

def write_to_sheet(df, tab):
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(tab)
        ws.clear()
    except Exception:
        ws = sh.add_worksheet(title=tab, rows=20000, cols=40)

    df = df.copy()
    for c in df.select_dtypes(include=["datetime64[ns]"]):
        df[c] = df[c].astype(str)
    df = df.fillna("")

    ws.update([df.columns.tolist()] + df.values.tolist())
    print(f"✅ Written {len(df)} rows to {tab}")

# =========================================
# MAIN
# =========================================

mapping  = get_adtopia_mapping()
all_data = []

for mcc in MCC_IDS:
    print(f"\nFetching MCC: {mcc}")
    df = fetch_ads_data(mcc)
    if not df.empty:
        all_data.append(df)

raw_df         = pd.concat(all_data, ignore_index=True)
raw_df["Date"] = pd.to_datetime(raw_df["Date"])

raw_df = raw_df.merge(
    mapping,
    left_on="Campaign_ID",
    right_on="ad_campaign_id",
    how="left"
)

raw_df["Campaign_Name"] = raw_df["adtopia_campaign_name"].fillna(raw_df["Campaign"])
raw_df.drop(columns=["ad_campaign_id", "adtopia_campaign_name"], inplace=True)

print("\nTotal rows:", len(raw_df))
write_to_sheet(raw_df, RAW_TAB)

# =========================================
# FORECAST
# =========================================

today = pd.Timestamp.now().normalize()
df    = raw_df.copy()
df    = df[df["Date"] < today]

df = df.groupby(["Account", "Date"]).agg(
    Cost        = ("Cost",        "sum"),
    Conversions = ("Conversions", "sum")
).reset_index()

df = df[df["Conversions"] > 0]
df["CPL"] = df["Cost"] / df["Conversions"]

reference_date = df["Date"].max()
cutoff         = reference_date - pd.Timedelta(days=59)
df             = df[df["Date"] >= cutoff]

last7_start = reference_date - pd.Timedelta(days=6)
prev7_start = reference_date - pd.Timedelta(days=13)
prev7_end   = reference_date - pd.Timedelta(days=7)
next7_end   = reference_date + pd.Timedelta(days=7)

results = []

for account in df["Account"].unique():
    acc = df[df["Account"] == account]
    if len(acc) < 14:
        continue

    prophet_df = acc[["Date", "CPL"]].rename(columns={"Date": "ds", "CPL": "y"})
    model      = Prophet(weekly_seasonality=True)
    model.fit(prophet_df)

    future   = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    forecast_future = forecast[
        (forecast["ds"] > reference_date) &
        (forecast["ds"] <= next7_end)
    ]

    last7 = acc[acc["Date"] >= last7_start]
    prev7 = acc[(acc["Date"] >= prev7_start) & (acc["Date"] <= prev7_end)]

    last7_cost = last7["Cost"].sum()
    last7_conv = last7["Conversions"].sum()
    prev7_cost = prev7["Cost"].sum()
    prev7_conv = prev7["Conversions"].sum()

    last7_cpl = last7_cost / last7_conv if last7_conv > 0 else None
    prev7_cpl = prev7_cost / prev7_conv if prev7_conv > 0 else None

    if last7_cpl and prev7_cpl:
        current_pct = ((last7_cpl - prev7_cpl) / prev7_cpl) * 100
        trend       = "Improving" if last7_cpl < prev7_cpl else "Declining"
    else:
        current_pct = None
        trend       = None

    forecast_avg_cpl = forecast_future["yhat"].mean()

    if last7_cpl:
        fcast_pct          = ((forecast_avg_cpl - last7_cpl) / last7_cpl) * 100
        forecast_direction = (
            "Improving" if fcast_pct < -8
            else "Declining" if fcast_pct > 8
            else "Stable"
        )
    else:
        fcast_pct          = None
        forecast_direction = None

    for _, row in forecast_future.iterrows():
        results.append({
            "Date":               row["ds"],
            "Account":            account,
            "Actual_CPL":         None,
            "Forecast_CPL":       row["yhat"],
            "Lower_CI":           row["yhat_lower"],
            "Upper_CI":           row["yhat_upper"],
            "Last7_CPL":          last7_cpl,
            "Prev7_CPL":          prev7_cpl,
            "Last7_Cost":         last7_cost,
            "Last7_Conversions":  last7_conv,
            "Trend":              trend,
            "Current_Pct":        current_pct,
            "Forecast_Avg_CPL":   forecast_avg_cpl,
            "Forecast_Pct":       fcast_pct,
            "Forecast_Direction": forecast_direction,
            "MAPE":               None,
            "Reliability":        None
        })

final_df = pd.DataFrame(results)
write_to_sheet(final_df, FORECAST_TAB)

print("\n🎉 Pipeline completed successfully")
