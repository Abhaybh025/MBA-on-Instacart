import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(page_title="Market Basket Dashboard", layout="wide")

# ----------------------------------------
# LOAD DATA (df_compare.csv)
# ----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("df_compare.csv")

df = load_data()

# ----------------------------------------
# LOAD INSTACART PARQUET DATA
# ----------------------------------------
@st.cache_data
def load_instacart():
    df = pd.read_parquet("order_products.parquet")

    # optional sampling for performance
    if len(df) > 500_000:
        df = df.sample(500_000, random_state=42)

    return df

insta = load_instacart()

# ----------------------------------------
# INSTACART DASHBOARD FUNCTION
# ----------------------------------------
def instacart_dashboard(df):
    st.title("📊 Instacart EDA Dashboard")

    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", int(df["order_id"].nunique()))
    col2.metric("Unique Products", int(df["product_id"].nunique()))
    col3.metric("Users", int(df["user_id"].nunique()))
    avg_days = df["days_since_prior_order"].dropna().mean() if "days_since_prior_order" in df.columns else None
    col4.metric("Avg Days Gap", round(avg_days, 2) if avg_days else "N/A")

    st.markdown("---")

    # Orders by Day of Week
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Orders by Day of Week")
        if "order_dow" in df.columns:
            dow = df["order_dow"].value_counts().sort_index().reset_index()
            dow.columns = ["order_dow", "count"]
            fig = px.bar(dow, x='order_dow', y='count')
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Orders by Hour")
        if "order_hour_of_day" in df.columns:
            hour = df["order_hour_of_day"].value_counts().sort_index().reset_index()
            hour.columns = ["hour", "count"]
            fig = px.line(hour, x='hour', y='count')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Departments & Aisles
    colC, colD = st.columns(2)

    with colC:
        st.subheader("Top 10 Departments")
        if "department" in df.columns:
            dept = df.groupby("department")["order_id"].count().sort_values(ascending=False).head(10).reset_index()
            dept.columns = ["department", "orders"]
            fig = px.bar(dept, y='department', x='orders', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

    with colD:
        st.subheader("Top 10 Aisles")
        if "aisle" in df.columns:
            aisle = df.groupby("aisle")["order_id"].count().sort_values(ascending=False).head(10).reset_index()
            aisle.columns = ["aisle", "orders"]
            fig = px.bar(aisle, y='aisle', x='orders', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    # ---------------------------------------------------------
    # 1️⃣ TOTAL ORDERS & REORDERS — AISLES + DEPARTMENTS
    # ---------------------------------------------------------
    st.subheader("📦 Orders & Reorders Across Aisles and Departments")

    col1, col2 = st.columns(2)

    with col1:
        if "aisle" in df.columns and "reordered" in df.columns:
            aisle_orders = df.groupby("aisle")["order_id"].count().sort_values(ascending=False).head(15)
            aisle_reorders = df.groupby("aisle")["reordered"].sum().sort_values(ascending=False).head(15)

            fig = px.bar(
                x=aisle_orders.index,
                y=aisle_orders.values,
                labels={"x": "Aisle", "y": "Count"},
                title="Top Aisles: Orders vs Reorders"
            )
            fig.add_bar(x=aisle_reorders.index, y=aisle_reorders.values, name="Reorders")
            fig.update_layout(xaxis_tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "department" in df.columns and "reordered" in df.columns:
            dept_orders = df.groupby("department")["order_id"].count().sort_values(ascending=False).head(15)
            dept_reorders = df.groupby("department")["reordered"].sum().sort_values(ascending=False).head(15)

            fig = px.bar(
                x=dept_orders.index,
                y=dept_orders.values,
                labels={"x": "Department", "y": "Count"},
                title="Top Departments: Orders vs Reorders"
            )
            fig.add_bar(x=dept_reorders.index, y=dept_reorders.values, name="Reorders")
            fig.update_layout(xaxis_tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 2️⃣ HIGHEST & LOWEST REORDER RATIO — AISLES + DEPARTMENTS
    # ---------------------------------------------------------
    st.subheader("♻️ Highest & Lowest Reorder Ratios")

    col3, col4 = st.columns(2)

    # Aisle reorder ratio
    with col3:
        if "aisle" in df.columns and "reordered" in df.columns:
            aisle_ratio = df.groupby("aisle")["reordered"].mean().sort_values(ascending=False)
            fig = px.bar(
                aisle_ratio.head(15),
                title="Top 15 Aisles • Highest Reorder Ratio"
            )
            fig.update_layout(xaxis_tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "aisle" in df.columns and "reordered" in df.columns:
            fig = px.bar(
                aisle_ratio.tail(15),
                title="Bottom 15 Aisles • Lowest Reorder Ratio"
            )
            fig.update_layout(xaxis_tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 3️⃣ MOST POPULAR PRODUCTS
    # ---------------------------------------------------------
    st.subheader("🏆 Most Popular Products")

    if "product_name" in df.columns:
        top_products = df.groupby("product_name")["order_id"].count().sort_values(ascending=False).head(20)
        fig = px.bar(
            x=top_products.index,
            y=top_products.values,
            labels={"x": "Product", "y": "Orders"},
            title="Top 20 Most Ordered Products"
        )
        fig.update_layout(xaxis_tickangle=90)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 4️⃣ CUMULATIVE UNIQUE USERS PER PRODUCT
    # ---------------------------------------------------------
    st.subheader("📈 Cumulative Unique Users per Product")

    if "user_id" in df.columns:
        user_product = df.groupby("product_name")["user_id"].nunique().sort_values(ascending=False).head(20)
        cumulative_users = user_product.cumsum()

        fig = px.line(
            x=cumulative_users.index,
            y=cumulative_users.values,
            labels={"x": "Product", "y": "Cumulative Users"},
            title="Cumulative Unique Users • Top 20 Products"
        )
        fig.update_layout(xaxis_tickangle=90)
        st.plotly_chart(fig, use_container_width=True)

    # ### 

    # st.subheader("6. Mean Reorder Ratio: Organic vs Inorganic")
    # organic_ratio = df.groupby("Organic")["reorder_ratio"].mean().reset_index()
    # fig7 = px.bar(organic_ratio, x="Organic", y="reorder_ratio",
    #               title="Mean Reorder Ratio (Organic vs Inorganic)")
    # st.plotly_chart(fig7, use_container_width=True)

    # ### 
    # st.subheader("7. Total Organic vs Inorganic Products")
    # organic_total = df["organic_flag"].value_counts().reset_index()
    # organic_total.columns = ["organic_flag", "count"]
    # fig8 = px.pie(organic_total, names="organic_flag", values="count",
    #               title="Organic vs Inorganic Product Distribution")
    # st.plotly_chart(fig8, use_container_width=True)

    ### 
    st.subheader(" Reorder Percentage vs Total Unique Users (Product Level)")
    prod_user_stats = df.groupby("product_name").agg(
        reorder_pct=("reordered", "mean"),
        unique_users=("user_id", "nunique")
    ).reset_index()

    fig9 = px.scatter(prod_user_stats,
                      x="unique_users",
                      y="reorder_pct",
                      trendline="ols",
                      title="Reorder % vs Unique Users per Product")
    st.plotly_chart(fig9, use_container_width=True)

    ### 
    st.subheader(" Reorder Percentage vs Total Orders (Product Level)")
    prod_order_stats = df.groupby("product_name").agg(
        reorder_pct=("reordered", "mean"),
        total_orders=("order_id", "count")
    ).reset_index()

    fig10 = px.scatter(prod_order_stats,
                       x="total_orders",
                       y="reorder_pct",
                       trendline="ols",
                       title="Reorder % vs Total Orders")
    st.plotly_chart(fig10, use_container_width=True)

    ### 
    st.subheader(" Most Popular Products Across Days of Week")
    dow_popular = (df.groupby(["order_dow", "product_name"])["order_id"]
                     .count()
                     .reset_index()
                     .sort_values(["order_dow", "order_id"], ascending=[True, False]))

    top_dow = dow_popular.groupby("order_dow").head(5)

    fig11 = px.bar(top_dow,
                   x="product_name",
                   y="order_id",
                   color="order_dow",
                   title="Top 5 Products for Each Day of Week")
    st.plotly_chart(fig11, use_container_width=True)

# ----------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Algorithm Comparison", "Itemset & Rules Analysis","Instacart Insights","Instacart Insights1", "Conclusions"]
)

# ----------------------------------------
# PAGE 1 — OVERVIEW
# ----------------------------------------
if page == "Overview":
    st.title("🛒 Market Basket Analysis Dashboard")
    st.subheader("Apriori vs FP-Growth Comparative Study")

    st.write(""" 

    #### Project Overview/Agenda
• To determine which association rule mining algorithm — Apriori or FP-Growth — demonstrates superior performance on large-scale, sparse retail transaction data.  
 • To identify and establish meaningful item relationships that can translate into actionable insights for product placement, bundling strategies, and recommendation systems in e-commerce and quick commerce environments.

#### Algorithm Comparison – Apriori vs FP-Growth
- **Execution Time**  
 • Apriori consistently ran far faster than FP-Growth across all support thresholds.
 • FP-Growth showed very high computation time (140–320 seconds), while Apriori finished within 3–8 seconds, making it significantly more efficient for this dataset.
- **Itemset Generation**  
 • Both algorithms produced similar numbers of frequent itemsets for all support values.
 • At lower support (0.003), both generated around 800+ itemsets, decreasing sharply as support increased.
 • This indicates that the dataset’s structure — large variety and sparse baskets — limits FP-Growth’s expected speed advantage.
- **Rule Count**  
 • The number of rules generated by both algorithms was nearly identical.
 • Lower support produced more rules, but overall rule volume remained moderate, showing that strong associations are relatively rare in Instacart’s diverse grocery baskets.
- **Key Insight**  
 Although FP-Growth is theoretically faster on dense datasets, our results show that Apriori outperforms FP-Growth in execution time without compromising itemset or rule quality.
 This makes Apriori the more practical choice for large, sparse retail datasets like Instacart.
""")

    st.write("""
    This dashboard visualizes and compares **Apriori** and **FP-Growth** algorithms  
    based on:
    - Runtime  
    - Number of Itemsets  
    - Number of Rules  
    - Effect of Support Threshold  
    """)

    st.write("### 📘 Summary Table")
    st.dataframe(df, use_container_width=True)



# ----------------------------------------
# PAGE 2 — ALGORITHM COMPARISON
# ----------------------------------------
elif page == "Algorithm Comparison":
    st.title("⚔️ Apriori vs FP-Growth — Comparison")

    st.write("### 🔍 Raw Comparison Table")
    st.dataframe(df, use_container_width=True)

    # Runtime Chart
    st.write("### ⏱️ Runtime Comparison")

    if st.toggle("Show visualization 1"):
        fig, ax = plt.subplots()
        ax.plot(df["support"], df["apriori_time"], marker="o", label="Apriori Time")
        ax.plot(df["support"], df["fpgrowth_time"], marker="o", label="FP-Growth Time")

        ax.set_xlabel("Support")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Runtime Comparison")
        ax.legend()
        st.pyplot(fig)

    # Itemset Count Comparision 
    st.write("### 📊 Itemset Count Comparison")

    if st.toggle("Show visualization 2"):
        fig, ax = plt.subplots()
        ax.bar(df['support']-0.0002, df['apriori_itemsets'], width=0.0004, label='Apriori')
        ax.bar(df['support']+0.0002, df['fpgrowth_itemsets'], width=0.0004, label='FP-Growth')
        ax.set_xlabel("Support")
        ax.set_ylabel("Number of Itemsets")
        ax.set_title("Apriori vs FP-Growth — Itemsets Count")
        ax.legend()
        st.pyplot(fig)


    # Itemsets VS Time
    st.write("### 🆚️ Itemsets vs Computation Time")

    if st.toggle("Show visualization 3"):
        fig, ax = plt.subplots()
        ax.scatter(df["apriori_itemsets"], df["apriori_time"])
        ax.set_xlabel("Apriori Itemsets")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title("Apriori: Itemsets vs Time")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.scatter(df["fpgrowth_itemsets"], df["fpgrowth_time"],color='orange')
        ax.set_xlabel("FP-growth Itemsets")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title("FP-growth: Itemsets vs Time")
        st.pyplot(fig)

    # Rules VS Time 

    st.write("### 🆚️ Rules vs Computation Time")

    if st.toggle("Show visualization 4"):
        fig, ax = plt.subplots()
        ax.scatter(df["apriori_rules"], df["apriori_time"])
        ax.set_xlabel("Apriori Rules")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title("Apriori: Rules Count vs Time")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.scatter(df["fpgrowth_rules"], df["fpgrowth_time"], color ='orange')
        ax.set_xlabel("FP-growth Rules")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title("FP-growth: Rules Count vs Time")
        st.pyplot(fig)



# ----------------------------------------
# PAGE 3 — RULES ANALYSIS
# ----------------------------------------
elif page == "Itemset & Rules Analysis":
    st.title("📊 Itemset & Rules Analysis")

    support_choice = st.selectbox(
        "Select Support Value",
        df['support'].unique()
    )

    selected = df[df['support'] == support_choice].iloc[0]


    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Apriori Itemsets", selected["apriori_itemsets"])
    col2.metric("FP-Growth Itemsets", selected["fpgrowth_itemsets"])
    col3.metric("Apriori Rules", selected["apriori_rules"])
    col4.metric("FP-Growth Rules", selected["fpgrowth_rules"])


    # Rules Count Visualization
    st.write("### 📘 Itemsets Generated")

    fig3, ax3 = plt.subplots()
    ax3.plot(df["support"], df["apriori_itemsets"], marker="o", label="Apriori Rules")
    ax3.plot(df["support"], df["fpgrowth_itemsets"], marker="o", label="FP-Growth Rules")

    ax3.set_xlabel("Support")
    ax3.set_ylabel("Rules")
    ax3.set_title("Rules vs Support")
    ax3.legend()
    st.pyplot(fig3)

    st.write("### 📘 Rules Generated")

    fig4, ax4 = plt.subplots()
    ax4.plot(df["support"], df["apriori_rules"], marker="o", label="Apriori Rules")
    ax4.plot(df["support"], df["fpgrowth_rules"], marker="o", label="FP-Growth Rules")

    ax4.set_xlabel("Support")
    ax4.set_ylabel("Rules")
    ax4.set_title("Rules vs Support")
    ax4.legend()
    st.pyplot(fig4)


    st.write("""
    ### Observations:
    - Lower support always → more itemsets & rules  
    - Apriori and FP-Growth generate **same rules/itemsets**  
    - FP-Growth runtime is extremely high due to wide dataset  
    """)

# ----------------------------------------
# PAGE 5 — CONCLUSIONS
# ----------------------------------------
elif page == "Conclusions":
    st.title("📌 Final Conclusions")

    st.write("""
    ### Key Findings  
    - Apriori performed **faster** than FP-Growth in your dataset  
    - FP-Growth slows down when number of columns (items) is high  
    - Apriori works well for:
      - moderate datasets  
      - binary matrix  
      - not too low support  

    ### Real-World Use Cases  
    - Retail recommendations  
    - Designing combo offers  
    - Store layout optimization  
    - Market basket insights for managers  

    ### Why This Dashboard Matters  
    - Helps analysts choose the right algorithm  
    - Clearly shows performance trade-offs  
    - Useful for academic research demonstrations  
    - Shows real benchmarking with real dataset  

    ### Future Enhancements  
    - Add interactive threshold selection  
    - Show top rules based on lift  
    - Integrate product names from products.csv  
    """)


# ----------------------------------------
# PAGE 4 — INSTACART EDA INSIGHTS
# ----------------------------------------
elif page == "Instacart Insights":

    st.title("🛒 Instacart Dataset Insights")

    st.write("These are additional EDA visuals added to your dashboard.")

    # Make sure dataset contains necessary columns
    st.write("Dataset Preview:")
    st.dataframe(insta.head(), use_container_width=True)

    # ---------------------------------------------------------
    # 1. TOTAL ORDERS & REORDERS FROM MOST POPULAR AISLES / DEPTS
    # ---------------------------------------------------------
    st.subheader("1️⃣ Total Orders & Reorders by Aisles and Departments")

    col1, col2 = st.columns(2)

    # Aisles
    aisle_orders = insta.groupby("aisle")["order_id"].count().sort_values(ascending=False).head(15)
    aisle_reorders = insta.groupby("aisle")["reordered"].sum().sort_values(ascending=False).head(15)

    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.bar(aisle_orders.index, aisle_orders.values, label="Orders")
    ax1.bar(aisle_reorders.index, aisle_reorders.values, label="Reorders")
    ax1.set_title("Top Aisles: Orders vs Reorders")
    ax1.tick_params(axis='x', rotation=90)
    ax1.legend()
    col1.pyplot(fig1)

    # Departments
    dept_orders = insta.groupby("department")["order_id"].count().sort_values(ascending=False)
    dept_reorders = insta.groupby("department")["reordered"].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.bar(dept_orders.index, dept_orders.values, label="Orders")
    ax2.bar(dept_reorders.index, dept_reorders.values, label="Reorders")
    ax2.set_title("Departments: Orders vs Reorders")
    ax2.tick_params(axis='x', rotation=90)
    ax2.legend()
    col2.pyplot(fig2)


    # ---------------------------------------------------------
    # 2. HIGHEST & LOWEST REORDER RATIO (AISLES / DEPARTMENTS)
    # ---------------------------------------------------------
    st.subheader("2️⃣ Aisles & Departments with Highest & Lowest Reorder Ratio")

    insta["reorder_ratio"] = insta["reordered"] / insta.groupby("product_id")["reordered"].transform("sum")

    col3, col4 = st.columns(2)

    # Highest aisles
    aisle_ratio = insta.groupby("aisle")["reordered"].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.bar(aisle_ratio.head(15).index, aisle_ratio.head(15).values)
    ax3.set_title("Top 15 Aisles – Highest Reorder Ratio")
    ax3.tick_params(axis='x', rotation=90)
    col3.pyplot(fig3)

    # Lowest aisles
    fig4, ax4 = plt.subplots(figsize=(8,5))
    ax4.bar(aisle_ratio.tail(15).index, aisle_ratio.tail(15).values)
    ax4.set_title("Bottom 15 Aisles – Lowest Reorder Ratio")
    ax4.tick_params(axis='x', rotation=90)
    col4.pyplot(fig4)


    # ---------------------------------------------
    # 3. MOST POPULAR PRODUCTS
    # ---------------------------------------------
    st.subheader("3️⃣ Most Popular Products (Highest Order Count)")

    top_products = insta.groupby("product_name")["order_id"].count().sort_values(ascending=False).head(20)

    fig5, ax5 = plt.subplots(figsize=(14,6))
    ax5.bar(top_products.index, top_products.values)
    ax5.set_title("Top 20 Most Ordered Products")
    ax5.tick_params(axis='x', rotation=90)
    st.pyplot(fig5)


    # ---------------------------------------------------------
    # 4. CUMULATIVE UNIQUE USERS PER PRODUCT
    # ---------------------------------------------------------
    st.subheader("4️⃣ Cumulative Unique Users per Product")

    user_product = insta.groupby("product_name")["user_id"].nunique().sort_values(ascending=False).head(20)
    cumulative_users = user_product.cumsum()

    fig6, ax6 = plt.subplots(figsize=(12,5))
    ax6.plot(cumulative_users.index, cumulative_users.values, marker="o")
    ax6.set_title("Cumulative Unique Users for Top Products")
    ax6.tick_params(axis='x', rotation=90)
    st.pyplot(fig6)



elif page == "Instacart Insights1":
    instacart_dashboard(insta)