import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from prophet import Prophet
import random

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
    df["StockCode"] = df["StockCode"].astype(str)  # фикс ошибки
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date

    return df


df = load_data()

# ---------------------------
# Категории
# ---------------------------
def get_category(desc):
    desc = str(desc).upper()

    if "MUG" in desc:
        return "Кружки"
    elif "CAKE" in desc:
        return "Выпечка"
    elif "BAG" in desc:
        return "Сумки"
    elif "LIGHT" in desc or "LAMP" in desc:
        return "Освещение"
    elif "CLOCK" in desc:
        return "Часы"
    elif "HEART" in desc or "STAR" in desc or "WOOD" in desc:
        return "Декор"
    elif "GIFT" in desc or "BOX" in desc:
        return "Подарки"
    else:
        return "Другое"

df["Category"] = df["Description"].apply(get_category)
product_categories = df.groupby("StockCode")["Category"].first()

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
    if max_score == 0:
        recs["Rating"] = 0
    else:
        recs["Rating"] = (recs["FinalScore"] / max_score) * 100

    recs["Rating"] = recs["Rating"].round(1)

    recs["Description"] = recs["StockCode"].map(product_names)
    recs["Category"] = recs["StockCode"].map(product_categories)

    recs = recs.sort_values("Rating", ascending=False)

    return recs[["StockCode", "Category", "Description", "Rating"]].head(num_recommendations)

# ---------------------------
# Cold start методы
# ---------------------------
def popular_products(n=5):
    popular = (
        df.groupby(["StockCode", "Category", "Description"])["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    popular.columns = ["StockCode", "Категория", "Товар", "Популярность"]
    return popular

def recommend_by_category(category, n=5):
    recs = (
        df[df["Category"] == category]
        .groupby(["StockCode", "Category", "Description"])["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    recs.columns = ["StockCode", "Категория", "Товар", "Популярность"]
    return recs

def recommend_by_interest(categories, n=5):
    recs = (
        df[df["Category"].isin(categories)]
        .groupby(["StockCode", "Category", "Description"])["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    recs.columns = ["StockCode", "Категория", "Товар", "Популярность"]
    return recs

# ---------------------------
# Метрики качества
# ---------------------------
def evaluate_models(sample_size=100, k=5):
    users = user_item_matrix.index.tolist()
    users = random.sample(users, min(sample_size, len(users)))

    hits_cf = 0
    hits_hybrid = 0
    precision_cf_total = 0
    precision_hybrid_total = 0
    recall_cf_total = 0
    recall_hybrid_total = 0
    total = 0

    for user in users:
        user_items = df[df["CustomerID"] == user]["StockCode"].unique()
        if len(user_items) < 2:
            continue

        test_item = random.choice(user_items)

        temp_matrix = user_item_matrix.copy()
        if test_item in temp_matrix.columns:
            temp_matrix.loc[user, test_item] = 0

        temp_similarity = cosine_similarity(temp_matrix)
        temp_similarity_df = pd.DataFrame(
            temp_similarity,
            index=temp_matrix.index,
            columns=temp_matrix.index
        )

        similar_users = temp_similarity_df[user].sort_values(ascending=False)
        similar_users = similar_users.drop(user)
        top_users = similar_users.head(5).index

        purchases = temp_matrix.loc[top_users]
        recs = purchases.sum().sort_values(ascending=False)

        cf_recs = recs.head(k).index.tolist()
        hybrid_recs = hybrid_recommend(user, k)["StockCode"].tolist()

        precision_cf_total += int(test_item in cf_recs) / k
        precision_hybrid_total += int(test_item in hybrid_recs) / k

        recall_cf_total += int(test_item in cf_recs)
        recall_hybrid_total += int(test_item in hybrid_recs)

        if test_item in cf_recs:
            hits_cf += 1
        if test_item in hybrid_recs:
            hits_hybrid += 1

        total += 1

    if total == 0:
        return None

    return {
        "precision_cf": precision_cf_total / total,
        "precision_hybrid": precision_hybrid_total / total,
        "recall_cf": recall_cf_total / total,
        "recall_hybrid": recall_hybrid_total / total,
        "hit_cf": hits_cf / total,
        "hit_hybrid": hits_hybrid / total,
    }

# ---------------------------
# Вкладки интерфейса
# ---------------------------
tab1, tab2 = st.tabs(["Основное приложение", "Админ-панель"])

# ---------------------------
# Основное приложение
# ---------------------------
with tab1:
    st.subheader("📊 Общая статистика")

    col1, col2, col3 = st.columns(3)
    col1.metric("Пользователи", df["CustomerID"].nunique())
    col2.metric("Товары", df["StockCode"].nunique())
    col3.metric("Транзакции", df["InvoiceNo"].nunique())

    st.subheader("📈 Продажи по дням")
    daily_sales = df.groupby("Date")["TotalPrice"].sum()

    fig, ax = plt.subplots()
    daily_sales.plot(ax=ax)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Продажи")
    st.pyplot(fig)

    st.subheader("🔮 Прогноз продаж на 30 дней")
    ts = daily_sales.reset_index()
    ts.columns = ["ds", "y"]

    model = Prophet()
    model.fit(ts)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig2 = model.plot(forecast)
    st.pyplot(fig2)

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
            categories = sorted(df["Category"].unique())
            selected_category = st.selectbox("Выберите группу товаров", categories)
            st.table(recommend_by_category(selected_category))

        else:
            categories = sorted(df["Category"].unique())
            selected_categories = st.multiselect(
                "Выберите интересующие группы",
                categories
            )

            if len(selected_categories) > 0:
                st.table(recommend_by_interest(selected_categories))

# ---------------------------
# Админ-панель
# ---------------------------
with tab2:
    st.header("⚙️ Админ-панель: оценка моделей")

    if st.button("Рассчитать метрики"):
        with st.spinner("Расчет..."):
            results = evaluate_models()

        st.subheader("Precision@5")
        st.metric("CF", round(results["precision_cf"], 3))
        st.metric("Hybrid", round(results["precision_hybrid"], 3))

        st.subheader("Recall@5")
        st.metric("CF", round(results["recall_cf"], 3))
        st.metric("Hybrid", round(results["recall_hybrid"], 3))

        st.subheader("HitRate@5")
        st.metric("CF", round(results["hit_cf"], 3))
        st.metric("Hybrid", round(results["hit_hybrid"], 3))

        if results["hit_hybrid"] > results["hit_cf"]:
            st.success("Гибридная модель показывает лучшие бизнес-результаты.")
        else:
            st.warning("Модели показывают сопоставимые результаты.")
