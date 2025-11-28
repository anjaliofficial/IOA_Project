# ============================ 1️⃣ IMPORTS ============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import make_interp_spline

sns.set(style="whitegrid")

# ============================ 2️⃣ LOAD DATA ============================
st.title("🎵 DataWave Music Churn Analysis Dashboard")
file = r'C:\\Users\\bista\\Documents\\Downloads\\DataWave_Music_Sprint_Dataset.csv'
df = pd.read_csv(file)

# ============================ 3️⃣ DATA CLEANING ============================
# Remove duplicates & null user_id
df = df.drop_duplicates(subset=['user_id']).dropna(subset=['user_id'])

# Gender
df = df.dropna(subset=['gender'])
df['gender'] = df['gender'].str.lower().str.strip().map({
    'f':'Female','female':'Female',
    'm':'Male','male':'Male',
    'other':'Other'
})

# Country & Region
df['country'] = df['country'].str.lower().str.strip().str.replace('.', '', regex=False)
country_map = {'uk':'United Kingdom','united kingdom':'United Kingdom','usa':'United States','ind':'India'}
df['country'] = df['country'].map(country_map).fillna(df['country']).str.title()
country_to_region = {'United States':'North America','United Kingdom':'Europe','India':'Asia'}
df['region'] = df['country'].map(country_to_region).fillna('Other')

# Developed vs Emerging Markets
developed_countries = ['United States','United Kingdom']
df['market_type'] = df['country'].apply(lambda x: 'Developed' if x in developed_countries else 'Emerging')

# Subscription Type
df['subscription_type'] = df['subscription_type'].str.lower().str.strip().map({
    'studnt':'Student','fam':'Family','premum':'Premium'
}).fillna(df['subscription_type']).str.title()

# Satisfaction Score
df = df.dropna(subset=['satisfaction_score'])

# Skip Rate
def clean_skip_rate(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    if "%" in x:
        x = x.replace("%","")
        return pd.to_numeric(x, errors="coerce") / 100
    num = pd.to_numeric(x, errors="coerce")
    if pd.notna(num): return num
    words = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    return words.get(x, np.nan)
df["skip_rate"] = df["skip_rate"].apply(clean_skip_rate)
df["skip_rate"] = df["skip_rate"].apply(lambda x: x/100 if pd.notna(x) and x>1 else x)

# Churn
df['churned'] = df['churned'].astype(str).str.capitalize().str.strip().map({'1':'Yes','0':'No'}).map({'No':0,'Yes':1})

# Monthly Fee
def clean_monthly_fee(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower().replace("usd","").replace("$","").strip()
    num = pd.to_numeric(x, errors='coerce')
    if pd.notna(num): return num
    if x in ["free","none","no fee"]: return 0.0
    return np.nan
df['monthly_fee_clean'] = df['monthly_fee'].apply(clean_monthly_fee)

# Join Date & Tenure
df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['join_date'])
today = pd.to_datetime('today')
df['tenure_months'] = (today.year - df['join_date'].dt.year)*12 + (today.month - df['join_date'].dt.month)

# Age Group
def age_group(age):
    if pd.isna(age): return 'Unknown'
    age = int(age)
    if age <= 18: return '13-18'
    elif age <= 24: return '19-24'
    elif age <= 34: return '25-34'
    elif age <= 44: return '35-44'
    elif age <= 54: return '45-54'
    elif age <= 64: return '55-64'
    else: return '65+'
df['age_group'] = df['age'].apply(age_group)

# ============================ 4️⃣ FILTERS ============================
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect("Select Region", options=df['region'].unique(), default=df['region'].unique())
selected_subscription = st.sidebar.multiselect("Select Subscription Type", options=df['subscription_type'].unique(), default=df['subscription_type'].unique())
selected_age_group = st.sidebar.multiselect("Select Age Group", options=df['age_group'].unique(), default=df['age_group'].unique())

df_filtered = df[(df['region'].isin(selected_region)) &
                 (df['subscription_type'].isin(selected_subscription)) &
                 (df['age_group'].isin(selected_age_group))]

# ============================ 5️⃣ OVERALL METRICS ============================
st.subheader("📊 Overall Metrics")
metrics = {
    'Overall Churn (%)': df_filtered['churned'].mean()*100,
    'Retention Rate (%)': 100 - df_filtered['churned'].mean()*100,
    'Avg Weekly Listening Hours': df_filtered['avg_listening_hours_per_week'].mean(),
    'Avg Satisfaction Score': df_filtered['satisfaction_score'].mean()
}
col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Churn", f"{metrics['Overall Churn (%)']:.2f}%")
col2.metric("Retention Rate", f"{metrics['Retention Rate (%)']:.2f}%")
col3.metric("Avg Weekly Listening", f"{metrics['Avg Weekly Listening Hours']:.2f}")
col4.metric("Avg Satisfaction", f"{metrics['Avg Satisfaction Score']:.2f}")

# ============================ 6️⃣ CHURN ANALYSIS ============================
st.subheader("Churn by Subscription Type")
sub_churn = df_filtered.groupby('subscription_type')['churned'].mean()*100
fig, ax = plt.subplots()
sns.barplot(x=sub_churn.index, y=sub_churn.values, palette='Set2', ax=ax)
ax.set_ylabel("Churn Rate (%)")
st.pyplot(fig)

st.subheader("Churn by Region")
region_churn = df_filtered.groupby('region')['churned'].mean()*100
fig, ax = plt.subplots()
sns.barplot(x=region_churn.index, y=region_churn.values, palette='Set1', ax=ax)
ax.set_ylabel("Churn Rate (%)")
st.pyplot(fig)

st.subheader("Churn by Age Group")
age_churn = df_filtered.groupby('age_group')['churned'].mean()*100
fig, ax = plt.subplots()
sns.barplot(x=age_churn.index, y=age_churn.values, palette='coolwarm', ax=ax)
ax.set_ylabel("Churn Rate (%)")
st.pyplot(fig)

# Heatmap: Age Group vs Subscription
st.subheader("Churn Heatmap: Age Group vs Subscription Type")
churn_table = df_filtered.groupby(['age_group','subscription_type'])['churned'].mean().unstack()*100
fig, ax = plt.subplots()
sns.heatmap(churn_table, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
st.pyplot(fig)

# Smooth Curve: Churn by Age Group
st.subheader("Smooth Curve: Churn Rate by Age Group")
age_order = ['13-18','19-24','25-34','35-44','45-54','55-64','65+']
age_churn = df_filtered.groupby('age_group')['churned'].mean().reindex(age_order)*100
age_churn = age_churn.dropna()
x_vals = range(len(age_churn))
x_smooth = np.linspace(0, len(age_churn)-1, 300)
spl = make_interp_spline(x_vals, age_churn.values, k=3)
y_smooth = spl(x_smooth)
fig, ax = plt.subplots()
ax.plot(x_smooth, y_smooth, color='red', marker='o')
ax.set_xticks(range(len(age_churn)))
ax.set_xticklabels(age_churn.index)
ax.set_ylabel("Churn Rate (%)")
st.pyplot(fig)

# ============================ 7️⃣ Satisfaction & Listening ============================
st.subheader("Satisfaction Score by Churn")
fig, ax = plt.subplots()
sns.boxplot(x='churned', y='satisfaction_score', data=df_filtered, palette='Set3', ax=ax)
st.pyplot(fig)

st.subheader("Distribution of Weekly Listening Hours")
fig, ax = plt.subplots()
sns.histplot(df_filtered['avg_listening_hours_per_week'], bins=30, kde=True, color='skyblue', ax=ax)
st.pyplot(fig)

st.subheader("Average Weekly Listening Hours by Churn")
fig, ax = plt.subplots()
sns.boxplot(x='churned', y='avg_listening_hours_per_week', data=df_filtered, palette='cool', ax=ax)
st.pyplot(fig)

st.subheader("Skip Rate vs Churn")
fig, ax = plt.subplots()
sns.scatterplot(x='skip_rate', y='churned', data=df_filtered, alpha=0.6, ax=ax)
st.pyplot(fig)

# ============================ 8️⃣ Developed vs Emerging Market ============================
st.subheader("Normalized Metrics: Developed vs Emerging Market")
metrics_list = ['churned','monthly_fee_clean','avg_listening_hours_per_week','satisfaction_score']
market_agg = df_filtered.groupby(['market_type','subscription_type'])[metrics_list].mean().reset_index()
for metric in metrics_list:
    min_val, max_val = market_agg[metric].min(), market_agg[metric].max()
    market_agg[metric] = (market_agg[metric]-min_val)/(max_val-min_val) if max_val!=min_val else 0.5
    heatmap_data = market_agg.pivot(index="market_type", columns="subscription_type", values=metric)
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title(f"Normalized {metric.replace('_',' ').title()} by Market & Subscription")
    st.pyplot(fig)

# ============================ 9️⃣ Logistic Regression & Risk Score ============================
numeric_cols = ['total_songs_played','satisfaction_score','monthly_fee_clean','age','avg_listening_hours_per_week','skip_rate']
X = df_filtered[numeric_cols].fillna(0)
y = df_filtered['churned'].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

coef_df = pd.DataFrame({'Feature': numeric_cols, 'Coefficient': logreg.coef_[0]}).sort_values(by='Coefficient', key=abs, ascending=False)
st.subheader("Logistic Regression Coefficients")
fig, ax = plt.subplots()
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm', ax=ax)
st.pyplot(fig)

df_filtered['churn_risk_score'] = logreg.predict_proba(X_scaled)[:,1]
risk_bins = pd.qcut(df_filtered['churn_risk_score'], q=5, labels=False)
risk_summary = df_filtered.groupby(risk_bins)['churned'].agg(['count','mean']).rename(columns={'mean':'actual_churn_rate'})
st.subheader("Actual Churn Rate by Predicted Risk Quintiles")
fig, ax = plt.subplots()
sns.barplot(x=risk_summary.index, y=risk_summary['actual_churn_rate']*100, palette='Reds', ax=ax)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Risk Quintile")
st.pyplot(fig)

# ============================ 1️⃣0️⃣ Cohort Analysis ============================
df_filtered['join_year'] = df_filtered['join_date'].dt.year
annual_cohorts = df_filtered.groupby('join_year').agg(users=('user_id','count'), churned=('churned','sum')).reset_index()
annual_cohorts['churn_rate'] = annual_cohorts['churned']/annual_cohorts['users']*100

st.subheader("Annual Cohort Churn Rate")
fig, ax = plt.subplots()
sns.barplot(x='join_year', y='churn_rate', data=annual_cohorts, palette='Blues', ax=ax)
ax.set_ylabel("Churn Rate (%)")
st.pyplot(fig)

# Quarterly Cohorts
df_filtered['join_year_quarter_str'] = df_filtered['join_date'].dt.to_period('Q').astype(str)
quarterly_cohorts = df_filtered.groupby('join_year_quarter_str').agg(users=('user_id','count'), churned=('churned','sum')).reset_index()
quarterly_cohorts['churn_rate'] = quarterly_cohorts['churned']/quarterly_cohorts['users']*100

st.subheader("Quarterly Cohort Churn Rate")
fig, ax = plt.subplots(figsize=(10,4))
sns.lineplot(x='join_year_quarter_str', y='churn_rate', data=quarterly_cohorts, marker='o', ax=ax)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Join Quarter")
plt.xticks(rotation=45)
st.pyplot(fig)

# Smooth curve for quarterly cohorts
x_vals = np.arange(len(quarterly_cohorts))
x_smooth = np.linspace(0, len(x_vals)-1, 300)
spl = make_interp_spline(x_vals, quarterly_cohorts['churn_rate'].values, k=3)
y_smooth = spl(x_smooth)
fig, ax = plt.subplots()
ax.plot(x_smooth, y_smooth, color='green', marker='o')
ax.set_xticks(range(len(quarterly_cohorts)))
ax.set_xticklabels(quarterly_cohorts['join_year_quarter_str'], rotation=45)
ax.set_ylabel("Churn Rate (%)")
st.pyplot(fig)

# ============================ 1️⃣1️⃣ Cognitive Bias ============================
st.subheader("Social Proof / Bandwagon Effect by Region")
region_group = df_filtered.groupby('region')['churned'].mean()
fig, ax = plt.subplots()
sns.barplot(x=region_group.index, y=region_group.values, palette='Set1', ax=ax)
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# Free-tier high engagement users
free_high_engagement = df_filtered[(df_filtered['monthly_fee_clean']==0) & (df_filtered['avg_listening_hours_per_week']>df_filtered['avg_listening_hours_per_week'].median())]
st.text(f"Free Tier High Engagement Users: {len(free_high_engagement)}")

# Anchoring Effect
early = df_filtered[df_filtered['join_date'].dt.year<2023]
late = df_filtered[df_filtered['join_date'].dt.year>=2023]
st.text(f"Anchoring Effect - Early Joiners Churn: {early['churned'].mean()*100:.2f}%")
st.text(f"Anchoring Effect - Late Joiners Churn: {late['churned'].mean()*100:.2f}%")

# ============================ END OF DASHBOARD ============================
st.success("✅ Dashboard Loaded Successfully!")
