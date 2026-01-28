import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest

st.set_page_config(page_title="AI Wastewater Management", layout="wide")

st.title("AI 기반 폐수 처리 관리 시스템")
st.caption("Las Vegas water-scarce city benchmarking")

# 데이터 로드
df = pd.read_csv("waste_water_treatment.csv")

# 전처리
df = df[['Variable', 'VariableDescription', 'Country', 'Year', 'PercentageValue']].dropna()
df['Year'] = df['Year'].astype(int)
df['PercentageValue'] = df['PercentageValue'].astype(float)

le_country = LabelEncoder()
le_var = LabelEncoder()

df['Country_enc'] = le_country.fit_transform(df['Country'])
df['Var_enc'] = le_var.fit_transform(df['Variable'])

# 모델 학습
X = df[['Country_enc', 'Var_enc', 'Year']]
y = df['PercentageValue']

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# 이상탐지 준비
df = df.sort_values(['Country', 'Variable', 'Year'])
df['delta'] = df.groupby(['Country', 'Variable'])['PercentageValue'].diff()
df_anom = df.dropna(subset=['delta'])

iso = IsolationForest(contamination=0.03, random_state=42)
df_anom['anomaly'] = iso.fit_predict(df_anom[['delta']])

# UI
country = st.selectbox("국가 선택", sorted(df['Country'].unique()))
variable = st.selectbox("지표 선택", sorted(df['Variable'].unique()))
year = st.slider("연도 선택", int(df['Year'].min()), int(df['Year'].max()))

if st.button("수질 상태 예측"):
    c = le_country.transform([country])[0]
    v = le_var.transform([variable])[0]
    pred = model.predict([[c, v, year]])[0]
    st.success(f"예측 Percentage Value: {pred:.2f}")

st.subheader("연도별 변화 추이 및 이상 탐지")

plot_df = df_anom[(df_anom['Country'] == country) & (df_anom['Variable'] == variable)]

fig, ax = plt.subplots()
ax.plot(plot_df['Year'], plot_df['PercentageValue'], label="PercentageValue")
ax.scatter(
    plot_df[plot_df['anomaly'] == -1]['Year'],
    plot_df[plot_df['anomaly'] == -1]['PercentageValue'],
    color='red', label='Anomaly'
)
ax.set_xlabel("Year")
ax.set_ylabel("PercentageValue")
ax.legend()
st.pyplot(fig)
