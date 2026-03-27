import os
import pandas as pd
import numpy as np
import gspread
from google.ads.googleads.client import GoogleAdsClient
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
import pytz
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
    "8567995305",
    "5419700447"
]

SKIPPABLE_STATUSES = {'CANCELED', 'CLOSED'}

end_date   = datetime.today()
start_date = end_date - timedelta(days=60)
START_STR  = start_date.strftime("%Y-%m-%d")
END_STR    = end_date.strftime("%Y-%m-%d")

# =========================================
# LOGGING
# =========================================

pipeline_logs = []

def log_message(level, message):
    """Add a log entry with GMT timestamp"""
    gmt_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
    pipeline_logs.append({
        "Timestamp": gmt_time,
        "Level": level,
        "Message": message
    })
    print(f"[{gmt_time}] {level}: {message}")

# =========================================
# QUERIES
# Fixed:
# - Removed metrics.impressions > 0 (was hiding historical paused/suspended data)
# - Removed campaign.status != 'REMOVED' (removed campaigns still have valid spend)
# - Added metrics.cost_micros > 0 (only pull rows where money was spent)
# =========================================
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
      AND metrics.cost_micros > 0
    ORDER BY segments.date DESC
"""

# Fixed:
# - Removed level = 1 filter (was missing suspended accounts with historical data)
# - Removed status = 'ENABLED' filter (suspended accounts like O|A GPT PCP 2 have valid data)
CHILDREN_QUERY = """
    SELECT
        customer_client.id,
        customer_client.descriptive_name,
        customer_client.manager,
        customer_client.status,
        customer_client.level
    FROM customer_client
"""

# =========================================
# GOOGLE ADS CLIENT
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

# =========================================
# GOOGLE ADS PULL
# =========================================
def pull_ads_data():
    log_message("INFO", f"Starting Google Ads data pull (Date range: {START_STR} to {END_STR})")
    log_message("INFO", f"Processing {len(MCC_IDS)} MCC accounts")
    
    all_rows = []
    summary  = []

    for mcc_id in MCC_IDS:
        log_message("INFO", f"Processing MCC: {mcc_id}")
        try:
            client  = get_ads_client(mcc_id)
            service = client.get_service("GoogleAdsService")

            # Get ALL children with no filters
            children = []
            response = service.search(customer_id=mcc_id, query=CHILDREN_QUERY)
            for row in response:
                c = row.customer_client
                # Skip the MCC itself (level 0) and manager accounts
                if c.level > 0 and not c.manager:
                    children.append({
                        "id":     str(c.id),
                        "name":   c.descriptive_name,
                        "status": c.status.name,
                        "level":  c.level
                    })

            log_message("INFO", f"MCC {mcc_id}: Found {len(children)} child accounts")

            if len(children) == 0:
                log_message("WARNING", f"MCC {mcc_id}: No child accounts found")
                summary.append({
                    "MCC": mcc_id, "Account": "N/A", "ID": "N/A",
                    "Rows": 0, "Status": "NO CHILDREN FOUND"
                })
                continue

            for child in children:
                cid    = child["id"]
                name   = child["name"]
                status = child["status"]

                # Skip CANCELED and CLOSED — Google blocks API access entirely
                if status in SKIPPABLE_STATUSES:
                    log_message("INFO", f"Account '{name}' ({cid}): Skipped [{status}]")
                    summary.append({
                        "MCC": mcc_id, "Account": name, "ID": cid,
                        "Rows": 0, "Status": f"SKIPPED ({status})"
                    })
                    continue

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
                    
                    log_message("INFO", f"Account '{name}' ({cid}): Retrieved {count} rows")
                    summary.append({
                        "MCC": mcc_id, "Account": name, "ID": cid,
                        "Rows": count, "Status": "OK"
                    })

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                    log_message("ERROR", f"Account '{name}' ({cid}): {error_msg}")
                    summary.append({
                        "MCC": mcc_id, "Account": name, "ID": cid,
                        "Rows": 0, "Status": f"FAILED: {type(e).__name__}"
                    })

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:100]}"
            log_message("ERROR", f"MCC {mcc_id}: {error_msg}")
            summary.append({
                "MCC": mcc_id, "Account": "N/A", "ID": "N/A",
                "Rows": 0, "Status": f"MCC ERROR: {type(e).__name__}"
            })

    df = pd.DataFrame(all_rows)
    log_message("INFO", f"Data pull complete: {len(df)} total rows retrieved")

    if not df.empty:
        account_summary = df.groupby(["MCC_ID", "Account_Name"])["Cost"].agg(["count","sum"]).round(2)
        log_message("INFO", f"Data retrieved from {len(account_summary)} accounts")

    return df, pd.DataFrame(summary)

# =========================================
# PROPHET FORECASTING (Updated to match Taboola structure)
# =========================================
def run_prophet_forecast(df):
    log_message("INFO", "Starting Prophet forecast generation")
    try:
        from prophet import Prophet
    except ImportError:
        log_message("ERROR", "Prophet library not available, skipping forecast")
        return pd.DataFrame()

    if df.empty:
        log_message("WARNING", "Empty dataframe provided, skipping forecast")
        return pd.DataFrame()

    df_agg = df.copy()
    df_agg["data_date"]    = pd.to_datetime(df_agg["Date"])
    df_agg["cost"]         = df_agg["Cost"]
    df_agg["conversions"]  = df_agg["Conversions"]
    df_agg["account_name"] = df_agg["Account_Name"]
    df_agg["account_id"]   = df_agg["Account_ID"]

    # AGGREGATE TO ACCOUNT LEVEL (matching Taboola structure)
    df_agg = df_agg.groupby(["data_date", "account_name", "account_id"]).agg(
        cost=("cost", "sum"),
        conversions=("conversions", "sum")
    ).reset_index()

    today          = pd.Timestamp.now().normalize()
    df_agg         = df_agg[df_agg["data_date"] < today]
    reference_date = df_agg["data_date"].max()

    cutoff = reference_date - pd.Timedelta(days=59)
    df_agg = df_agg[df_agg["data_date"] >= cutoff]

    df_raw      = df_agg.copy()
    df_filtered = df_agg[df_agg["conversions"] > 0].copy()
    df_filtered["cpl"] = df_filtered["cost"] / df_filtered["conversions"]

    log_message("INFO", f"Reference date: {reference_date.date()}, Filtered data: {df_filtered.shape[0]} rows")

    last7_start = reference_date - pd.Timedelta(days=6)
    prev7_start = reference_date - pd.Timedelta(days=13)
    prev7_end   = reference_date - pd.Timedelta(days=7)
    next7_end   = reference_date + pd.Timedelta(days=7)

    def fit_prophet_series(daily_df, value_col, log_transform=True):
        pdf = daily_df[["data_date", value_col]].rename(columns={"data_date": "ds", value_col: "y"})
        pdf = pdf[pdf["y"] > 0].copy()
        if log_transform: 
            pdf["y"] = np.log(pdf["y"])
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.8
        )
        model.fit(pdf)
        forecast = model.predict(model.make_future_dataframe(periods=7))
        if log_transform:
            for c in ["yhat", "yhat_lower", "yhat_upper"]: 
                forecast[c] = np.exp(forecast[c])
        return forecast

    def run_prophet(daily_df, kpi_df):
        daily_df = daily_df.sort_values("data_date").copy()
        daily_df["cpl"] = daily_df["cost"] / daily_df["conversions"]

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

        # Forecast cost and conversions separately (new)
        cost_f = fit_prophet_series(daily_df, "cost")
        conv_f = fit_prophet_series(daily_df, "conversions")

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
            "Last7_CPL":          round(last7_cpl, 4) if last7_cpl else None,
            "Prev7_CPL":          round(prev7_cpl, 4) if prev7_cpl else None,
            "Last7_Cost":         round(last7_cost, 2),
            "Last7_Conversions":  int(last7_conv),
            "Prev7_Cost":         round(prev7_cost, 2),
            "Prev7_Conversions":  int(prev7_conv),
            "Trend":              trend,
            "Current_Pct":        round(current_pct, 2) if current_pct else None,
            "Forecast_Avg_CPL":   round(forecast_avg_cpl, 4),
            "Forecast_Pct":       round(fcast_pct, 2) if fcast_pct else None,
            "Forecast_Direction": forecast_direction,
            "MAPE":               round(mape, 2) if not np.isnan(mape) else None,
            "Reliability":        reliability,
        }

        # Return cost and conv forecasts as well
        return forecast_future, kpis, cost_f[cost_f["ds"] > reference_date], conv_f[conv_f["ds"] > reference_date]

    results  = []
    accounts = df_filtered[["account_name", "account_id"]].drop_duplicates()
    
    log_message("INFO", f"Generating forecasts for {len(accounts)} accounts")

    for _, acct_row in accounts.iterrows():
        account = acct_row["account_name"]
        aid = acct_row["account_id"]
        account_df     = df_filtered[df_filtered["account_id"] == aid].copy()
        account_kpi_df = df_raw[df_raw["account_id"] == aid].copy()
        if len(account_df) < 14:
            log_message("WARNING", f"Account '{account}' ({aid}): Insufficient data ({len(account_df)} days), skipping forecast")
            continue

        try:
            forecast_future, kpis, cost_forecast, conv_forecast = run_prophet(account_df, account_kpi_df)
            
            # Create lookup dictionaries for cost and conv forecasts
            cost_map = cost_forecast.set_index("ds")["yhat"].to_dict()
            conv_map = conv_forecast.set_index("ds")["yhat"].to_dict()
            
            # Get the last date for conditional KPI population
            last_ds = forecast_future["ds"].max()
            
            for _, row in forecast_future.iterrows():
                is_last = (row["ds"] == last_ds)
                results.append({
                    "Date":               str(row["ds"].date()),
                    "Account":            account,
                    "Account_ID":         aid,
                    "Actual_CPL":         None,
                    "Actual_Spend":       None,
                    "Actual_Conversions": None,
                    "Forecast_CPL":       round(row["yhat"], 4),
                    "Lower_CI":           round(row["yhat_lower"], 4),
                    "Upper_CI":           round(row["yhat_upper"], 4),
                    "Forecast_Cost":      round(cost_map.get(row["ds"], 0), 2),
                    "Forecast_Conv":      round(conv_map.get(row["ds"], 0), 2),
                    "Last7_CPL":          kpis["Last7_CPL"] if is_last else None,
                    "Prev7_CPL":          kpis["Prev7_CPL"] if is_last else None,
                    "Last7_Cost":         kpis["Last7_Cost"] if is_last else None,
                    "Last7_Conversions":  kpis["Last7_Conversions"] if is_last else None,
                    "Prev7_Cost":         kpis["Prev7_Cost"] if is_last else None,
                    "Prev7_Conversions":  kpis["Prev7_Conversions"] if is_last else None,
                    "Trend":              kpis["Trend"] if is_last else None,
                    "Current_Pct":        kpis["Current_Pct"] if is_last else None,
                    "Forecast_Avg_CPL":   kpis["Forecast_Avg_CPL"] if is_last else None,
                    "Forecast_Pct":       kpis["Forecast_Pct"] if is_last else None,
                    "Forecast_Direction": kpis["Forecast_Direction"] if is_last else None,
                    "MAPE":               kpis["MAPE"] if is_last else None,
                    "Reliability":        kpis["Reliability"] if is_last else None,
                })
            log_message("INFO", f"Account '{account}' ({aid}): Forecast generated successfully")
        except Exception as e:
            log_message("ERROR", f"Account '{account}' ({aid}): Forecast failed - {str(e)[:100]}")
            continue

    log_message("INFO", "Forecasting: All Accounts Combined")
    portfolio_daily     = df_filtered.groupby("data_date").agg(
        cost=("cost", "sum"), conversions=("conversions", "sum")).reset_index()
    portfolio_daily_raw = df_raw.groupby("data_date").agg(
        cost=("cost", "sum"), conversions=("conversions", "sum")).reset_index()

    try:
        forecast_future, kpis, cost_forecast, conv_forecast = run_prophet(portfolio_daily, portfolio_daily_raw)
        
        cost_map = cost_forecast.set_index("ds")["yhat"].to_dict()
        conv_map = conv_forecast.set_index("ds")["yhat"].to_dict()
        last_ds = forecast_future["ds"].max()
        
        for _, row in forecast_future.iterrows():
            is_last = (row["ds"] == last_ds)
            results.append({
                "Date":               str(row["ds"].date()),
                "Account":            "All Accounts Combined",
                "Account_ID":         "ALL",
                "Actual_CPL":         None,
                "Actual_Spend":       None,
                "Actual_Conversions": None,
                "Forecast_CPL":       round(row["yhat"], 4),
                "Lower_CI":           round(row["yhat_lower"], 4),
                "Upper_CI":           round(row["yhat_upper"], 4),
                "Forecast_Cost":      round(cost_map.get(row["ds"], 0), 2),
                "Forecast_Conv":      round(conv_map.get(row["ds"], 0), 2),
                "Last7_CPL":          kpis["Last7_CPL"] if is_last else None,
                "Prev7_CPL":          kpis["Prev7_CPL"] if is_last else None,
                "Last7_Cost":         kpis["Last7_Cost"] if is_last else None,
                "Last7_Conversions":  kpis["Last7_Conversions"] if is_last else None,
                "Prev7_Cost":         kpis["Prev7_Cost"] if is_last else None,
                "Prev7_Conversions":  kpis["Prev7_Conversions"] if is_last else None,
                "Trend":              kpis["Trend"] if is_last else None,
                "Current_Pct":        kpis["Current_Pct"] if is_last else None,
                "Forecast_Avg_CPL":   kpis["Forecast_Avg_CPL"] if is_last else None,
                "Forecast_Pct":       kpis["Forecast_Pct"] if is_last else None,
                "Forecast_Direction": kpis["Forecast_Direction"] if is_last else None,
                "MAPE":               kpis["MAPE"] if is_last else None,
                "Reliability":        kpis["Reliability"] if is_last else None,
            })
        log_message("INFO", f"Combined forecast generated - CPL: ${kpis['Forecast_Avg_CPL']:.2f}, Trend: {kpis['Trend']}")
    except Exception as e:
        log_message("ERROR", f"Combined forecast failed: {str(e)[:100]}")

    # Build historical actual data table with new columns
    actual_table = df_filtered.copy()
    actual_table = actual_table.rename(columns={
        "data_date":    "Date",
        "account_name": "Account",
        "account_id":   "Account_ID",
        "cost":         "Actual_Spend",
        "conversions":  "Actual_Conversions",
        "cpl":          "Actual_CPL"
    })
    actual_table["Date"] = actual_table["Date"].astype(str)
    for col in ["Forecast_CPL", "Lower_CI", "Upper_CI", "Forecast_Cost", "Forecast_Conv",
                "Last7_CPL", "Prev7_CPL", "Last7_Cost", "Last7_Conversions",
                "Prev7_Cost", "Prev7_Conversions", "Trend", "Current_Pct",
                "Forecast_Avg_CPL", "Forecast_Pct", "Forecast_Direction",
                "MAPE", "Reliability"]:
        actual_table[col] = None

    # Portfolio historical with new columns
    portfolio_hist = portfolio_daily.copy()
    portfolio_hist["Account"]            = "All Accounts Combined"
    portfolio_hist["Account_ID"]         = "ALL"
    portfolio_hist["Actual_CPL"]         = portfolio_hist["cost"] / portfolio_hist["conversions"]
    portfolio_hist["Actual_Spend"]       = portfolio_hist["cost"]
    portfolio_hist["Actual_Conversions"] = portfolio_hist["conversions"]
    portfolio_hist["Date"]               = portfolio_hist["data_date"].astype(str)
    for col in ["Forecast_CPL", "Lower_CI", "Upper_CI", "Forecast_Cost", "Forecast_Conv",
                "Last7_CPL", "Prev7_CPL", "Last7_Cost", "Last7_Conversions",
                "Prev7_Cost", "Prev7_Conversions", "Trend", "Current_Pct",
                "Forecast_Avg_CPL", "Forecast_Pct", "Forecast_Direction",
                "MAPE", "Reliability"]:
        portfolio_hist[col] = None
    portfolio_hist = portfolio_hist.drop(columns=["cost", "conversions", "data_date"])

    forecast_df = pd.DataFrame(results)
    final_table = pd.concat([actual_table, portfolio_hist, forecast_df], ignore_index=True)
    final_table = final_table.sort_values(["Account", "Date"]).reset_index(drop=True)

    # Updated column order to match Taboola structure
    col_order = [
        "Date", "Account", "Account_ID", "Actual_CPL", "Actual_Spend", "Actual_Conversions",
        "Forecast_CPL", "Lower_CI", "Upper_CI", "Forecast_Cost", "Forecast_Conv",
        "Last7_CPL", "Prev7_CPL", "Last7_Cost", "Last7_Conversions", 
        "Prev7_Cost", "Prev7_Conversions",
        "Trend", "Current_Pct", "Forecast_Avg_CPL", "Forecast_Pct",
        "Forecast_Direction", "MAPE", "Reliability"
    ]
    final_table = final_table[col_order]
    final_table = final_table.fillna("")

    log_message("INFO", f"Forecast table generated: {len(final_table)} rows")
    return final_table

# =========================================
# GOOGLE SHEETS WRITE
# =========================================
def get_sheets_client():
    log_message("INFO", "Authenticating with Google Sheets")
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
    log_message("INFO", "Google Sheets authentication successful")
    return gspread.authorize(creds)

def write_to_sheet(sh, tab_name, df):
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
        log_message("INFO", f"Sheet '{tab_name}': Cleared existing data")
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=200000, cols=25)
        log_message("INFO", f"Sheet '{tab_name}': Created new worksheet")

    if len(df) > 0:
        headers = list(df.columns)
        data    = df.fillna("").values.tolist()
        ws.update([headers])
        for i in range(0, len(data), 5000):
            ws.append_rows(data[i:i+5000], value_input_option="USER_ENTERED")
        log_message("INFO", f"Sheet '{tab_name}': Wrote {len(df)} rows")
    else:
        log_message("WARNING", f"Sheet '{tab_name}': No data to write")

def write_logs_to_sheet(sh):
    """Write logs to sheet, keeping only last 3000 rows"""
    try:
        try:
            ws = sh.worksheet("Logs")
            # Read existing logs
            existing_data = ws.get_all_records()
            existing_logs = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
        except:
            ws = sh.add_worksheet(title="Logs", rows=3500, cols=3)
            existing_logs = pd.DataFrame()
            log_message("INFO", "Created new 'Logs' worksheet")
        
        # Combine existing and new logs
        new_logs = pd.DataFrame(pipeline_logs)
        
        if not existing_logs.empty and not new_logs.empty:
            # Ensure columns match
            if set(new_logs.columns) == set(existing_logs.columns):
                combined_logs = pd.concat([existing_logs, new_logs], ignore_index=True)
            else:
                combined_logs = new_logs
        elif not new_logs.empty:
            combined_logs = new_logs
        else:
            combined_logs = existing_logs
        
        # Keep only last 3000 rows
        if len(combined_logs) > 3000:
            combined_logs = combined_logs.tail(3000).reset_index(drop=True)
        
        # Write to sheet
        ws.clear()
        ws.update([list(combined_logs.columns)])
        data = combined_logs.fillna("").values.tolist()
        ws.append_rows(data, value_input_option="USER_ENTERED")
        
        print(f"[{datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}] INFO: Logs sheet updated with {len(combined_logs)} total rows (last 3000 kept)")
        
    except Exception as e:
        print(f"[{datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to write logs to sheet - {str(e)}")

# =========================================
# MAIN
# =========================================
def main():
    log_message("INFO", "Pipeline execution started")
    
    try:
        # Step 1: Pull Google Ads data
        ads_df, df_summary = pull_ads_data()

        # Step 2: Run Prophet forecast
        forecast_df = run_prophet_forecast(ads_df)

        # Step 3: Write to Google Sheets
        log_message("INFO", "Connecting to Google Sheets")
        gc = get_sheets_client()
        sh = gc.open_by_key(SHEET_ID)
        log_message("INFO", f"Connected to spreadsheet: {sh.title}")

        write_to_sheet(sh, "AdsData", ads_df)

        if not forecast_df.empty:
            write_to_sheet(sh, "Forecast", forecast_df)
        else:
            log_message("WARNING", "No forecast data generated")

        # Write logs last
        write_logs_to_sheet(sh)
        
        log_message("INFO", "Pipeline execution completed successfully")
        print("\nAll done! ✓")
        
    except Exception as e:
        log_message("ERROR", f"Pipeline execution failed: {str(e)}")
        # Still try to write logs even if pipeline fails
        try:
            gc = get_sheets_client()
            sh = gc.open_by_key(SHEET_ID)
            write_logs_to_sheet(sh)
        except:
            pass
        raise

if __name__ == "__main__":
    main()
