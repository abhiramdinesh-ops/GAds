import os
import pandas as pd
import gspread
from google.ads.googleads.client import GoogleAdsClient
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from datetime import datetime, timedelta

# Google Ads credentials (115251666473 client)
ADS_CLIENT_ID       = os.environ["ADS_CLIENT_ID"]
ADS_CLIENT_SECRET   = os.environ["ADS_CLIENT_SECRET"]
ADS_REFRESH_TOKEN   = os.environ["ADS_REFRESH_TOKEN"]
DEVELOPER_TOKEN     = os.environ["DEVELOPER_TOKEN"]

# Google Sheets credentials (770621812681 client)
SHEETS_CLIENT_ID     = os.environ["SHEETS_CLIENT_ID"]
SHEETS_CLIENT_SECRET = os.environ["SHEETS_CLIENT_SECRET"]
SHEETS_REFRESH_TOKEN = os.environ["SHEETS_REFRESH_TOKEN"]

SHEET_ID = os.environ["SHEET_ID"]

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

def get_ads_client(mcc_id):
    return GoogleAdsClient.load_from_dict({
        "developer_token":   DEVELOPER_TOKEN,
        "client_id":         ADS_CLIENT_ID,
        "client_secret":     ADS_CLIENT_SECRET,
        "refresh_token":     ADS_REFRESH_TOKEN,
        "login_customer_id": mcc_id,
        "use_proto_plus":    True,
    })

def main():
    print(f"Starting: {datetime.now()}")
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
    print(f"\nTotal rows: {len(df)}")

    print("Writing to Google Sheet...")
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
    print("Sheets auth successful.")

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    print(f"Connected to: {sh.title}")

    try:
        ws = sh.worksheet("AdsData")
        ws.clear()
    except Exception:
        ws = sh.add_worksheet(title="AdsData", rows=200000, cols=20)

    if len(df) > 0:
        headers = list(df.columns)
        data    = df.fillna("").values.tolist()
        ws.update([headers])
        for i in range(0, len(data), 5000):
            ws.append_rows(data[i:i+5000], value_input_option="USER_ENTERED")
            print(f"  Written rows {i+1} to {min(i+5000, len(data))}")
        print(f"Total {len(df)} rows written.")
    else:
        print("No data to write.")

    try:
        log = sh.worksheet("RunLog")
    except Exception:
        log = sh.add_worksheet(title="RunLog", rows=1000, cols=6)
        log.append_row(["Timestamp", "Rows", "MCCs", "Accounts", "Status"])

    log.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(df),
        ", ".join(MCC_IDS),
        len(set(df["Account_ID"].tolist())) if len(df) > 0 else 0,
        "Success" if len(df) > 0 else "No data"
    ])
    print("RunLog updated. Done.")

if __name__ == "__main__":
    main()
