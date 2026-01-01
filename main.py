import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DOI Monitor", layout="wide")

# ===============================
# CITY MAPPINGS
# ===============================
SALES_CITY_MAPPING = {
    "Agra": "Agra",
    "Ahmedabad Rural": "Ahmedabad-Gandhinagar",
    "Ahmedabad-Gandhinagar": "Ahmedabad-Gandhinagar",
    "Allahabad": "Allahabad",
    "Bangalore": "Bangalore",
    "Bangalore Rural": "Bangalore",
    "Bhopal": "Bhopal",
    "Bhubaneshwar-Cuttack": "Bhubaneshwar-Cuttack",
    "Bhubaneswar Rural": "Bhubaneshwar-Cuttack",
    "Chandigarh Tricity": "Chandigarh Tricity",
    "Chennai": "Chennai",
    "Chennai Rural": "Chennai",
    "Coimbatore": "Coimbatore",
    "DehraDun": "DehraDun",
    "Gurgaon": "Delhi",
    "Gurugram Rural": "Delhi",
    "Guwahati": "Guwahati",
    "Hyderabad": "Hyderabad",
    "Hyderabad Rural": "Hyderabad",
    "Indore": "Indore",
    "Kochi": "Kochi",
    "Kochi Rural": "Kochi",
    "Kolkata": "Kolkata",
    "Kolkata Rural": "Kolkata",
    "Kozhikode": "Kozhikode",
    "Lucknow Rural": "Lucknow-Kanpur",
    "Lucknow-Kanpur": "Lucknow-Kanpur",
    "Mangaluru": "Mangaluru",
    "Mumbai": "Mumbai",
    "Mumbai Rural": "Mumbai",
    "Mysore": "Mysore",
    "Nagpur": "Nagpur",
    "Nashik": "Nashik",
    "Noida": "Noida",
    "Noida Rural": "Noida",
    "Patna": "Patna",
    "Patna Rural": "Patna",
    "Pune": "Pune",
    "Pune Rural": "Pune",
    "Raipur": "Raipur",
    "Ranchi": "Ranchi",
    "Ranchi Rural": "Ranchi",
    "Surat": "Surat",
    "Thiruvananthapuram": "Thiruvananthapuram",
    "Vadodara": "Vadodara",
    "Vijayawada-Guntur": "Vijayawada-Guntur",
    "Visakhapatnam": "Visakhapatnam",
    "Vizag Rural": "Visakhapatnam"
}

INVENTORY_CITY_MAPPING = {
    "Agra": "Agra",
    "Ahmedabad-Gandhinagar": "Ahmedabad-Gandhinagar",
    "Allahabad": "Allahabad",
    "Bangalore": "Bangalore",
    "Bhopal": "Bhopal",
    "Bhubaneshwar-Cuttack": "Bhubaneshwar-Cuttack",
    "Chandigarh Tricity": "Chandigarh Tricity",
    "Chennai": "Chennai",
    "Coimbatore": "Coimbatore",
    "DehraDun": "DehraDun",
    "Delhi": "Delhi",
    "Gurgaon": "Delhi",
    "Guwahati": "Guwahati",
    "Gwalior": "Gwalior",
    "Hyderabad": "Hyderabad",
    "Indore": "Indore",
    "Kochi": "Kochi",
    "Kolkata": "Kolkata",
    "Kozhikode": "Kozhikode",
    "Lucknow-Kanpur": "Lucknow-Kanpur",
    "Ludhiana": "Ludhiana",
    "Mangaluru": "Mangaluru",
    "Mumbai": "Mumbai",
    "Mysore": "Mysore",
    "Nagpur": "Nagpur",
    "Nashik": "Nashik",
    "Noida": "Noida",
    "Patna": "Patna",
    "Pune": "Pune",
    "Raipur": "Raipur",
    "Ranchi": "Ranchi",
    "Surat": "Surat",
    "Thiruvananthapuram": "Thiruvananthapuram",
    "Vadodara": "Vadodara",
    "Vijayawada-Guntur": "Vijayawada-Guntur",
    "Visakhapatnam": "Visakhapatnam"
}

# ===============================
# PREPROCESSING
# ===============================
@st.cache_data
def preprocess_data(uploaded_file):
    sales_df = pd.read_excel(uploaded_file, sheet_name="Sales", engine="openpyxl")
    inventory_df = pd.read_excel(uploaded_file, sheet_name="Inventory", engine="openpyxl")

    # ---- Sales preprocessing ----
    sales_df["source_city_name"] = sales_df["source_city_name"].replace(SALES_CITY_MAPPING)
    sales_df["date_range"] = pd.to_datetime(sales_df["date_range"])

    sales_processed = (
        sales_df
        .groupby(
            ["date_range", "source_city_name", "source_sku_id", "sku_description"],
            as_index=False
        )
        .agg({"total_quantity": "sum"})
        .rename(columns={
            "source_city_name": "city",
            "source_sku_id": "sku_id",
            "total_quantity": "sales_qty"
        })
    )

    # ---- Inventory preprocessing ----
    inventory_df["city"] = inventory_df["city"].replace(INVENTORY_CITY_MAPPING)

    inventory_processed = (
        inventory_df
        .groupby(["city", "sku_id", "sku_description"], as_index=False)
        .agg({"soh": "sum"})
        .rename(columns={"soh": "inventory_units"})
    )

    return sales_processed, inventory_processed

# ===============================
# DOI BASE FILE CREATION
# ===============================
@st.cache_data
def create_doi_base(sales_df, inventory_df, n_days):
    max_date = sales_df["date_range"].max()
    start_date = max_date - pd.Timedelta(days=n_days - 1)

    filtered_sales = sales_df[
        (sales_df["date_range"] >= start_date) &
        (sales_df["date_range"] <= max_date)
    ]

    sales_n_days = (
        filtered_sales
        .groupby(["sku_id", "city"], as_index=False)
        .agg({
            "sales_qty": "sum",
            "sku_description": "first"
        })
    )

    final_df = pd.merge(
        inventory_df,
        sales_n_days,
        on=["sku_id", "city"],
        how="outer",
        suffixes=("_inv", "_sales")
    )

    final_df["sku_description"] = (
        final_df["sku_description_inv"]
        .combine_first(final_df["sku_description_sales"])
    )

    final_df["sales_qty"] = final_df["sales_qty"].fillna(0)
    final_df["inventory_units"] = final_df["inventory_units"].fillna(0)

    return final_df[[
        "city", "sku_id", "sku_description", "inventory_units", "sales_qty"
    ]]

# ===============================
# DOI CALCULATION
# ===============================
def calculate_doi(df, days):
    df = df.copy()

    df["doi"] = np.where(
        df["inventory_units"] == 0,
        0,
        np.where(
            df["sales_qty"] == 0,
            df["inventory_units"],
            df["inventory_units"] / (df["sales_qty"] / days)
        )
    )

    df["doi"] = np.floor(df["doi"]).astype(int)
    return df

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📦 DOI Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Base Excel File",
    type=["xlsx"]
)

days = st.sidebar.number_input(
    "Number of days",
    min_value=1,
    value=7
)

# ===============================
# MAIN FLOW
# ===============================
if uploaded_file is None:
    st.info("👈 Upload the base Excel file to start")
    st.stop()

sales_df, inventory_df = preprocess_data(uploaded_file)
base_df = create_doi_base(sales_df, inventory_df, days)

# ===============================
# SESSION STATE
# ===============================
for key in ["pan_mode", "selected_sku", "selected_city"]:
    if key not in st.session_state:
        st.session_state[key] = "None"

def reset_individual():
    st.session_state.selected_sku = "None"
    st.session_state.selected_city = "None"

def reset_pan():
    st.session_state.pan_mode = "None"

# ===============================
# FILTERS
# ===============================
st.sidebar.selectbox(
    "Pan India View",
    ["None", "Product Wise", "City Wise"],
    key="pan_mode",
    on_change=reset_individual
)

sku_list = ["None"] + sorted(base_df["sku_description"].dropna().unique())
city_list = ["None"] + sorted(base_df["city"].dropna().unique())

st.sidebar.selectbox(
    "Individual SKU",
    sku_list,
    key="selected_sku",
    on_change=reset_pan
)

st.sidebar.selectbox(
    "Individual City",
    city_list,
    key="selected_city",
    on_change=reset_pan
)

# ===============================
# DATA PROCESSING (PRIORITY DRIVEN)
# ===============================
result_df = pd.DataFrame()

sku = st.session_state.selected_sku
city = st.session_state.selected_city
mode = st.session_state.pan_mode

# 1️⃣ SKU + CITY
if sku != "None" and city != "None":
    temp = base_df[
        (base_df["sku_description"] == sku) &
        (base_df["city"] == city)
    ].groupby(
        ["sku_description", "city"],
        as_index=False
    ).agg({
        "sales_qty": "sum",
        "inventory_units": "sum"
    })

    result_df = calculate_doi(temp, days)

# 2️⃣ INDIVIDUAL SKU
elif sku != "None":
    temp = base_df[
        base_df["sku_description"] == sku
    ].groupby(
        ["sku_description", "city"],
        as_index=False
    ).agg({
        "sales_qty": "sum",
        "inventory_units": "sum"
    })

    result_df = calculate_doi(temp, days)

# 3️⃣ INDIVIDUAL CITY
elif city != "None":
    temp = base_df[
        base_df["city"] == city
    ].groupby(
        ["city", "sku_description"],
        as_index=False
    ).agg({
        "sales_qty": "sum",
        "inventory_units": "sum"
    })

    result_df = calculate_doi(temp, days)

# 4️⃣ PAN INDIA – PRODUCT
elif mode == "Product Wise":
    temp = base_df.groupby(
        ["sku_description"],
        as_index=False
    ).agg({
        "sales_qty": "sum",
        "inventory_units": "sum"
    })

    result_df = calculate_doi(temp, days)

# 5️⃣ PAN INDIA – CITY
elif mode == "City Wise":
    temp = base_df.groupby(
        ["city"],
        as_index=False
    ).agg({
        "sales_qty": "sum",
        "inventory_units": "sum"
    })

    result_df = calculate_doi(temp, days)

# ===============================
# DISPLAY
# ===============================
st.title("📦 DOI Monitor")

if result_df.empty:
    st.info("Please select a Pan India view or an Individual SKU / City.")
else:
    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


