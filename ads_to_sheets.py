import os
import pandas as pd
import numpy as np
import gspread
from google.ads.googleads.client import GoogleAdsClient
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# =========================================
# CREDENTIALS
# =========================================
ADS_CLIENT_ID        = os.environ["ADS_CLIENT_ID"]
ADS_CLIENT_SECRET    = os.environ["ADS_CLIENT_SECRET"]
ADS_REFRESH_TOKEN    = os.environ["ADS_REFRESH_TOKEN"]
DEVELOPER_TOKEN      = os.environ["DEVELOPER_TOKEN"]

SHEETS_CLIENT_ID     = os.environ["SHEETS_CLIENT_ID"]
SHEETS_CLIENT_SECRET = os.environ["SHEETS_CLIENT_SECRET"]
SHEETS_REFRESH_TOKEN = os.environ["SHEETS_REFRESH_TOKEN"]

SHEET_ID = os.environ["SHEET_ID"]

# =========================================
# CONFIG
# =========================================
MCC_IDS = [
    "7141208780",
    "7309803413",
    "5419872903",
    "8567995305"
]

end_date   = datetime.today()
start_date = end_date - timedelta(days=60)
START_STR  = start_date.strftime("%Y-%m-%d")
END_STR    = end_date.strftime("%Y-%m-%d")

QUERY = f"""
    SELECT
        campaign.id,
        campaign.name,
        campaign.status,
        campaign.advertising_channel_type,
        segments.date,
        metrics.impressions,
        metrics.clicks,
        metrics.cost_micros,
        metrics.conversions,
        metrics.all_conversions_value,
        metrics.ctr,
        metrics.average_cpc,
        metrics.search_impression_share,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{START_STR}' AND '{END_STR}'
      AND metrics.impressions > 0
      AND campaign.status != 'REMOVED'
    ORDER BY segments.date DESC
"""

CHILDREN_QUERY = """
    SELECT
        customer_client.id,
        customer_client.descriptive_name,
        customer_client.manager,
        customer_client.status,
        customer_client.level
    FROM customer_client
    WHERE customer_client.level = 1
      AND customer_client.status = 'ENABLED'
"""

# =========================================
# GOOGLE ADS PULL
# =========================================
def get_ads_client(mcc_id):
    return GoogleAdsClient.load_from_dict({
        "developer_token":   DEVELOPER_TOKEN,
        "client_id":         ADS_CLIENT_ID,
        "client_secret":     ADS_CLIENT_SECRET,
        "refresh_token":     ADS_REFRESH_TOKEN,
        "login_customer_id": mcc_id,
        "use_proto_plus":    True,
    })

def pull_ads_data():
    print(f"Starting Ads pull: {datetime.now()}")
    print(f"Date range: {START_STR} to {END_STR}")
    all_rows = []

    for mcc_id in MCC_IDS:
        print(f"\nMCC: {mcc_id}")
        try:
            client  = get_ads_client(mcc_id)
            service = client.get_service("GoogleAdsService")

            children = []
            response = service.search(customer_id=mcc_id, query=CHILDREN_QUERY)
            for row in response:
                if not row.customer_client.manager:
                    children.append({
                        "id":   str(row.customer_client.id),
                        "name": row.customer_client.descriptive_name
                    })
            print(f"  Found {len(children)} child accounts")

            for child in children:
                cid  = child["id"]
                name = child["name"]
                print(f"  -> {name} ({cid})", end=" ")
                try:
                    resp  = service.search(customer_id=cid, query=QUERY)
                    count = 0
                    for row in resp:
                        m          = row.metrics
                        cost       = m.cost_micros / 1_000_000
                        conv_value = m.all_conversions_value
                        all_rows.append({
                            "MCC_ID":        mcc_id,
                            "Account_ID":    cid,
                            "Account_Name":  name,
                            "Campaign_ID":   str(row.campaign.id),
                            "Campaign_Name": row.campaign.name,
                            "Status":        row.campaign.status.name,
                            "Channel_Type":  row.campaign.advertising_channel_type.name,
                            "Date":          row.segments.date,
                            "Impressions":   m.impressions,
                            "Clicks":        m.clicks,
                            "Cost":          round(cost, 2),
                            "Conversions":   round(m.conversions, 2),
                            "Conv_Value":    round(conv_value, 2),
                            "CTR_Pct":       round(m.ctr * 100, 2),
                            "Avg_CPC":       round(m.average_cpc / 1_000_000, 2),
                            "Search_IS_Pct": round(m.search_impression_share * 100, 2),
                            "All_Conv":      round(m.all_conversions, 2),
                            "ROAS":          round(conv_value / cost, 2) if cost > 0 else 0
                        })
                        count += 1
                    print(f"- {count} rows")
                except Exception as e:
                    print(f"- SKIPPED: {e}")

        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows fetched: {len(df)}")
    return df

# =========================================
# PROPHET FORECASTING
# =========================================
def run_prophet_forecast(df):
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed, skipping forecast.")
        return pd.DataFrame()

    if df.empty:
        print("No data for forecasting.")
        return pd.DataFrame()

    # Aggregate to campaign + date level
    df_agg = df.copy()
    df_agg["data_date"]     = pd.to_datetime(df_agg["Date"])
    df_agg["cost"]          = df_agg["Cost"]
    df_agg["conversions"]   = df_agg["Conversions"]
    df_agg["campaign_name"] = df_agg["Campaign_Name"]

    df_agg = df_agg.groupby(["data_date", "campaign_name"]).agg(
        cost=("cost", "sum"),
        conversions=("conversions", "sum")
    ).reset_index()

    # Exclude today (partial data)
    today          = pd.Timestamp.now().normalize()
    df_agg         = df_agg[df_agg["data_date"] < today]
    reference_date = df_agg["data_date"].max()

    # Keep last 60 days
    cutoff = reference_date - pd.Timedelta(days=59)
    df_agg = df_agg[df_agg["data_date"] >= cutoff]

    # Raw for KPIs (includes zero conversion days)
    df_raw = df_agg.copy()

    # Filter for Prophet training
    df_filtered = df_agg[df_agg["conversions"] > 0].copy()
    df_filtered["cpl"] = df_filtered["cost"] / df_filtered["conversions"]

    print(f"\nReference date: {reference_date.date()}")
    print(f"After cleaning: {df_filtered.shape}")

    # Date windows
    last7_start = reference_date - pd.Timedelta(days=6)
    prev7_start = reference_date - pd.Timedelta(days=13)
    prev7_end   = reference_date - pd.Timedelta(days=7)
    next7_end   = reference_date + pd.Timedelta(days=7)

    def run_prophet(daily_df, kpi_df):
        daily_df = daily_df.sort_values("data_date").copy()
        daily_df["cpl"] = daily_df["cost"] / daily_df["conversions"]

        # IQR capping
        Q1    = daily_df["cpl"].quantile(0.25)
        Q3    = daily_df["cpl"].quantile(0.75)
        IQR   = Q3 - Q1
        lower = max(Q1 - 1.5 * IQR, 0.01)
        upper = Q3 + 1.5 * IQR
        daily_df["cpl_capped"] = daily_df["cpl"].clip(lower, upper)

        prophet_df = daily_df[["data_date", "cpl_capped"]].rename(
            columns={"data_date": "ds", "cpl_capped": "y"})
        prophet_df["y"] = np.log(prophet_df["y"])

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.8,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=5
        )
        model.fit(prophet_df)

        future   = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        forecast["yhat"]       = np.exp(forecast["yhat"])
        forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
        forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

        # MAPE on last 7 days
        actual_window = daily_df[daily_df["data_date"] >= last7_start]
        forecast_hist = forecast[
            (forecast["ds"] >= last7_start) &
            (forecast["ds"] <= reference_date)
        ]
        merged = actual_window[["data_date", "cpl"]].merge(
            forecast_hist[["ds", "yhat"]].rename(columns={"ds": "data_date"}),
            on="data_date", how="inner")
        if len(merged) > 0:
            mape = (np.abs(merged["cpl"] - merged["yhat"]) / merged["cpl"]).mean() * 100
            mape = min(mape, 999)
        else:
            mape = np.nan

        if np.isnan(mape):    reliability = "Unknown"
        elif mape < 15:       reliability = "High"
        elif mape < 35:       reliability = "Medium"
        else:                 reliability = "Low"

        # KPIs from raw data
        kpi_df     = kpi_df.sort_values("data_date").copy()
        last7      = kpi_df[kpi_df["data_date"] >= last7_start]
        prev7      = kpi_df[
            (kpi_df["data_date"] >= prev7_start) &
            (kpi_df["data_date"] <= prev7_end)
        ]

        last7_cost = last7["cost"].sum()
        last7_conv = last7["conversions"].sum()
        prev7_cost = prev7["cost"].sum()
        prev7_conv = prev7["conversions"].sum()

        last7_cpl = last7_cost / last7_conv if last7_conv > 0 else None
        prev7_cpl = prev7_cost / prev7_conv if prev7_conv > 0 else None

        if last7_cpl and prev7_cpl:
            current_pct = ((last7_cpl - prev7_cpl) / prev7_cpl) * 100
            if last7_cpl < prev7_cpl:    trend = "Improving"
            elif last7_cpl > prev7_cpl:  trend = "Declining"
            else:                         trend = "Stable"
        else:
            current_pct = None
            trend       = None

        forecast_future  = forecast[
            (forecast["ds"] > reference_date) &
            (forecast["ds"] <= next7_end)
        ]
        forecast_avg_cpl = forecast_future["yhat"].mean()

        if last7_cpl:
            fcast_pct = ((forecast_avg_cpl - last7_cpl) / last7_cpl) * 100
            if fcast_pct < -8:    forecast_direction = "Improving"
            elif fcast_pct > 8:   forecast_direction = "Declining"
            else:                  forecast_direction = "Stable"
        else:
            fcast_pct          = None
            forecast_direction = None

        kpis = {
            "Last7_CPL":          last7_cpl,
            "Prev7_CPL":          prev7_cpl,
            "Last7_Cost":         round(last7_cost, 2),
            "Last7_Conversions":  int(last7_conv),
            "Trend":              trend,
            "Current_Pct":        round(current_pct, 2) if current_pct else None,
            "Forecast_Avg_CPL":   round(forecast_avg_cpl, 4),
            "Forecast_Pct":       round(fcast_pct, 2) if fcast_pct else None,
            "Forecast_Direction": forecast_direction,
            "MAPE":               round(mape, 2) if not np.isnan(mape) else None,
            "Reliability":        reliability,
        }

        return forecast_future, kpis

    # =========================================
    # RUN PER CAMPAIGN
    # =========================================
    results   = []
    campaigns = df_filtered["campaign_name"].unique()

    for campaign in campaigns:
        campaign_df     = df_filtered[df_filtered["campaign_name"] == campaign].copy()
        campaign_kpi_df = df_raw[df_raw["campaign_name"] == campaign].copy()
        if len(campaign_df) < 14:
            print(f"Skipping {campaign} - not enough data ({len(campaign_df)} days)")
            continue

        print(f"Forecasting: {campaign}")
        try:
            forecast_future, kpis = run_prophet(campaign_df, campaign_kpi_df)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        for _, row in forecast_future.iterrows():
            results.append({
                "Date":               str(row["ds"].date()),
                "Account":            campaign,
                "Actual_CPL":         None,
                "Forecast_CPL":       round(row["yhat"], 4),
                "Lower_CI":           round(row["yhat_lower"], 4),
                "Upper_CI":           round(row["yhat_upper"], 4),
                "Last7_CPL":          kpis["Last7_CPL"],
                "Prev7_CPL":          kpis["Prev7_CPL"],
                "Last7_Cost":         kpis["Last7_Cost"],
                "Last7_Conversions":  kpis["Last7_Conversions"],
                "Trend":              kpis["Trend"],
                "Current_Pct":        kpis["Current_Pct"],
                "Forecast_Avg_CPL":   kpis["Forecast_Avg_CPL"],
                "Forecast_Pct":       kpis["Forecast_Pct"],
                "Forecast_Direction": kpis["Forecast_Direction"],
                "MAPE":               kpis["MAPE"],
                "Reliability":        kpis["Reliability"],
            })

    # =========================================
    # RUN ALL CAMPAIGNS COMBINED
    # =========================================
    print("Forecasting: All Campaigns Combined")
    portfolio_daily     = df_filtered.groupby("data_date").agg(
        cost=("cost", "sum"), conversions=("conversions", "sum")).reset_index()
    portfolio_daily_raw = df_raw.groupby("data_date").agg(
        cost=("cost", "sum"), conversions=("conversions", "sum")).reset_index()

    try:
        forecast_future, kpis = run_prophet(portfolio_daily, portfolio_daily_raw)
        for _, row in forecast_future.iterrows():
            results.append({
                "Date":               str(row["ds"].date()),
                "Account":            "All Campaigns Combined",
                "Actual_CPL":         None,
                "Forecast_CPL":       round(row["yhat"], 4),
                "Lower_CI":           round(row["yhat_lower"], 4),
                "Upper_CI":           round(row["yhat_upper"], 4),
                "Last7_CPL":          kpis["Last7_CPL"],
                "Prev7_CPL":          kpis["Prev7_CPL"],
                "Last7_Cost":         kpis["Last7_Cost"],
                "Last7_Conversions":  kpis["Last7_Conversions"],
                "Trend":              kpis["Trend"],
                "Current_Pct":        kpis["Current_Pct"],
                "Forecast_Avg_CPL":   kpis["Forecast_Avg_CPL"],
                "Forecast_Pct":       kpis["Forecast_Pct"],
                "Forecast_Direction": kpis["Forecast_Direction"],
                "MAPE":               kpis["MAPE"],
                "Reliability":        kpis["Reliability"],
            })
        print(f"  Portfolio forecast CPL: ${kpis['Forecast_Avg_CPL']:.2f}")
        print(f"  Portfolio last 7 CPL:   ${kpis['Last7_CPL']:.2f}")
        print(f"  Portfolio trend:        {kpis['Trend']}")
    except Exception as e:
        print(f"  Portfolio forecast error: {e}")

    # =========================================
    # HISTORICAL ACTUALS — per campaign
    # =========================================
    actual_table = df_filtered[["data_date", "campaign_name", "cpl"]].copy()
    actual_table = actual_table.rename(columns={
        "data_date":     "Date",
        "campaign_name": "Account",
        "cpl":           "Actual_CPL"
    })
    actual_table["Date"] = actual_table["Date"].astype(str)
    for col in ["Forecast_CPL", "Lower_CI", "Upper_CI", "Last7_CPL", "Prev7_CPL",
                "Last7_Cost", "Last7_Conversions", "Trend", "Current_Pct",
                "Forecast_Avg_CPL", "Forecast_Pct", "Forecast_Direction",
                "MAPE", "Reliability"]:
        actual_table[col] = None

    # =========================================
    # HISTORICAL ACTUALS — all combined
    # =========================================
    portfolio_hist = portfolio_daily.copy()
    portfolio_hist["Account"]    = "All Campaigns Combined"
    portfolio_hist["Actual_CPL"] = portfolio_hist["cost"] / portfolio_hist["conversions"]
    portfolio_hist["Date"]       = portfolio_hist["data_date"].astype(str)
    for col in ["Forecast_CPL", "Lower_CI", "Upper_CI", "Last7_CPL", "Prev7_CPL",
                "Last7_Cost", "Last7_Conversions", "Trend", "Current_Pct",
                "Forecast_Avg_CPL", "Forecast_Pct", "Forecast_Direction",
                "MAPE", "Reliability"]:
        portfolio_hist[col] = None
    portfolio_hist = portfolio_hist.drop(columns=["cost", "conversions", "data_date"])

    # =========================================
    # COMBINE + FINAL COLUMN ORDER
    # =========================================
    forecast_df = pd.DataFrame(results)
    final_table = pd.concat([actual_table, portfolio_hist, forecast_df], ignore_index=True)
    final_table = final_table.sort_values(["Account", "Date"]).reset_index(drop=True)

    # Enforce exact column order matching Power BI schema
    col_order = [
        "Date", "Account", "Actual_CPL", "Forecast_CPL", "Lower_CI", "Upper_CI",
        "Last7_CPL", "Prev7_CPL", "Last7_Cost", "Last7_Conversions",
        "Trend", "Current_Pct", "Forecast_Avg_CPL", "Forecast_Pct",
        "Forecast_Direction", "MAPE", "Reliability"
    ]
    final_table = final_table[col_order]
    final_table = final_table.fillna("")

    print(f"\nForecast table rows: {len(final_table)}")
    return final_table

# =========================================
# GOOGLE SHEETS WRITE
# =========================================
def get_sheets_client():
    creds = Credentials(
        token=None,
        refresh_token=SHEETS_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=SHEETS_CLIENT_ID,
        client_secret=SHEETS_CLIENT_SECRET,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    creds.refresh(Request())
    return gspread.authorize(creds)

def write_to_sheet(sh, tab_name, df):
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
        print(f"Cleared existing {tab_name} tab")
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=200000, cols=25)
        print(f"Created new {tab_name} tab")

    if len(df) > 0:
        headers = list(df.columns)
        data    = df.fillna("").values.tolist()
        ws.update([headers])
        for i in range(0, len(data), 5000):
            ws.append_rows(data[i:i+5000], value_input_option="USER_ENTERED")
            print(f"  {tab_name}: written rows {i+1} to {min(i+5000, len(data))}")
        print(f"  {tab_name}: {len(df)} total rows written")
    else:
        print(f"  {tab_name}: no data to write")

# =========================================
# MAIN
# =========================================
def main():
    # Step 1: Pull Google Ads data
    ads_df = pull_ads_data()

    # Step 2: Run Prophet forecast
    print("\nRunning Prophet forecast...")
    forecast_df = run_prophet_forecast(ads_df)

    # Step 3: Write to Google Sheets
    print("\nConnecting to Google Sheets...")
    gc = get_sheets_client()
    sh = gc.open_by_key(SHEET_ID)
    print(f"Connected to: {sh.title}")

    write_to_sheet(sh, "AdsData", ads_df)

    if not forecast_df.empty:
        write_to_sheet(sh, "Forecast", forecast_df)

    # RunLog
    try:
        log = sh.worksheet("RunLog")
    except Exception:
        log = sh.add_worksheet(title="RunLog", rows=1000, cols=7)
        log.append_row(["Timestamp", "Ads Rows", "Forecast Rows", "MCCs", "Accounts", "Status"])

    log.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(ads_df),
        len(forecast_df),
        ", ".join(MCC_IDS),
        len(set(ads_df["Account_ID"].tolist())) if len(ads_df) > 0 else 0,
        "Success"
    ])
    print("RunLog updated. All done.")

if __name__ == "__main__":
    main()
