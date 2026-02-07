import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from prophet import Prophet

st.set_page_config(page_title="Retail Recommendation System", layout="wide")

st.title("🛒 Retail Recommendation System")
st.write("Прогноз продаж и персональные рекомендации товаров")

# ---------------------------
# Загрузка и очистка данных
# ---------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date

    return df

df = load_data()

# ---------------------------
# Категории (для cold start)
# ---------------------------
def get_category(desc):
    desc = str(desc).upper()
    if "MUG" in desc:
        return "Кружки"
    elif "CAKE" in desc:
        return "Выпечка"
    elif "BAG" in desc:
        return "Сумки"
    elif "LIGHT" in desc:
        return "Освещение"
    elif "CLOCK" in desc:
        return "Часы"
    else:
        return "Другое"

df["Category"] = df["Description"].apply(get_category)

# ---------------------------
# Общая статистика
# ---------------------------
st.subheader("📊 Общая статистика")

col1, col2, col3 = st.columns(3)
col1.metric("Пользователи", df["CustomerID"].nunique())
col2.metric("Товары", df["StockCode"].nunique())
col3.metric("Транзакции", df["InvoiceNo"].nunique())

# ---------------------------
# Временной ряд
# ---------------------------
st.subheader("📈 Продажи по дням")

daily_sales = df.groupby("Date")["TotalPrice"].sum()

fig, ax = plt.subplots()
daily_sales.plot(ax=ax)
ax.set_xlabel("Дата")
ax.set_ylabel("Продажи")
st.pyplot(fig)

# ---------------------------
# Прогноз продаж
# ---------------------------
st.subheader("🔮 Прогноз продаж на 30 дней")

ts = daily_sales.reset_index()
ts.columns = ["ds", "y"]

model = Prophet()
model.fit(ts)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig2 = model.plot(forecast)
st.pyplot(fig2)

# ---------------------------
# Рекомендательная система
# ---------------------------
user_item_matrix = df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# тренд товаров (30 дней)
last_date = df["InvoiceDate"].max()
start_date = last_date - pd.Timedelta(days=30)

recent_sales = df[df["InvoiceDate"] >= start_date]
product_trend = (
    recent_sales.groupby("StockCode")["Quantity"]
    .sum()
)

product_names = (
    df.groupby("StockCode")["Description"]
    .first()
)

def recommend_products(customer_id, num_recommendations=50):
    similar_users = user_similarity_df[customer_id].sort_values(ascending=False)
    similar_users = similar_users.drop(customer_id)

    top_users = similar_users.head(5).index
    similar_users_purchases = user_item_matrix.loc[top_users]

    recommended_products = similar_users_purchases.sum().sort_values(ascending=False)

    user_purchases = user_item_matrix.loc[customer_id]
    already_bought = user_purchases[user_purchases > 0].index

    recommended_products = recommended_products.drop(already_bought, errors="ignore")

    return recommended_products.head(num_recommendations)

def hybrid_recommend(customer_id, num_recommendations=5):
    recs = recommend_products(customer_id)

    recs = recs.reset_index()
    recs.columns = ["StockCode", "Score"]

    recs["Trend"] = recs["StockCode"].map(product_trend).fillna(1)
    recs["FinalScore"] = recs["Score"] * recs["Trend"]

    max_score = recs["FinalScore"].max()
    recs["Rating"] = (recs["FinalScore"] / max_score) * 100

    recs["Description"] = recs["StockCode"].map(product_names)

    return recs[["Description", "Rating"]].head(num_recommendations)

# ---------------------------
# Cold start методы
# ---------------------------
def popular_products(n=5):
    popular = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    popular.columns = ["Description", "Popularity"]
    return popular

def recommend_by_category(category, n=5):
    recs = (
        df[df["Category"] == category]
        .groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    recs.columns = ["Description", "Popularity"]
    return recs

# ---------------------------
# Интерфейс рекомендаций
# ---------------------------
st.subheader("🛍 Рекомендации")

user_type = st.radio(
    "Тип пользователя",
    ["Существующий пользователь", "Новый пользователь"]
)

if user_type == "Существующий пользователь":
    customers = user_item_matrix.index.tolist()
    selected_user = st.selectbox("Выберите пользователя", customers)

    if st.button("Получить рекомендации"):
        recs = hybrid_recommend(selected_user)
        st.table(recs)

else:
    method = st.radio(
        "Выберите способ рекомендаций",
        ["Популярные товары", "По категории", "По интересам"]
    )

    if method == "Популярные товары":
        st.table(popular_products())

    elif method == "По категории":
        categories = df["Category"].unique()
        selected_category = st.selectbox("Выберите категорию", categories)
        st.table(recommend_by_category(selected_category))

    else:
        popular = popular_products(20)
        choices = st.multiselect(
            "Выберите товары, которые вам нравятся",
            popular["Description"].tolist()
        )

        if len(choices) > 0:
            recs = (
                df[df["Description"].isin(choices)]
                .groupby("Description")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            st.table(recs)
