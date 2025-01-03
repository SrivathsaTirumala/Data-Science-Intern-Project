# This code is a user interface for the code onlineretail.ipynb
# This code is completely optional or additional for this project
# to run the code, use the below command
# python3 -m streamlit run [file_name].py
# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit Configuration
st.set_page_config(page_title="Online Retail Recommendation System", layout="wide")
st.title("🛒 Online Retail Recommendation System")

# Function to load and clean data
@st.cache_data
def load_data(file_path):
    # Load the dataset
    df = pd.read_excel(file_path)
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    return df

# Function to display top products and countries
def show_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    # Top 10 Products
    st.write("**Top 10 Most Sold Products**")
    top_products = df['Description'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax)
    plt.title("Top 10 Products Sold")
    st.pyplot(fig)

    # Top 10 Countries
    st.write("**Top 10 Countries by Sales**")
    top_countries = df['Country'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma', ax=ax)
    plt.title("Top 10 Countries by Sales")
    st.pyplot(fig)

# Popularity-Based Recommendation
def popularity_recommendation(df):
    st.subheader("🏆 Popularity-Based Recommendation")
    popular_products = df.groupby('Description')['Quantity'].sum().reset_index()
    popular_products = popular_products.sort_values(by='Quantity', ascending=False).head(10)
    st.write(popular_products)

# Collaborative Filtering Recommendation
def collaborative_recommendation(df, customer_id):
    st.subheader("🤝 Collaborative Filtering Recommendation")

    # User-Item Matrix
    pivot_table = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
    sparse_matrix = csr_matrix(pivot_table)
    similarity_matrix = cosine_similarity(sparse_matrix)

    # Function to recommend products
    def recommend_products(customer_id, pivot_table, similarity_matrix, num_recommendations=5):
        if customer_id not in pivot_table.index:
            return ["Customer ID not found!"]

        customer_idx = pivot_table.index.get_loc(customer_id)
        sim_scores = similarity_matrix[customer_idx]
        similar_customers = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[1:6]

        recommended_products = set()
        for idx, score in similar_customers:
            similar_customer_id = pivot_table.index[idx]
            purchased_products = pivot_table.columns[pivot_table.loc[similar_customer_id] > 0].tolist()
            recommended_products.update(purchased_products)

        purchased_by_customer = pivot_table.columns[pivot_table.loc[customer_id] > 0].tolist()
        final_recommendations = [product for product in recommended_products if product not in purchased_by_customer]
        return final_recommendations[:num_recommendations]

    # Input for Customer ID
    recommendations = recommend_products(customer_id, pivot_table, similarity_matrix)
    st.write(f"Recommended Products for Customer ID {customer_id}:")
    st.write(recommendations)

# Content-Based Recommendation
def content_based_recommendation(df, product_id):
    st.subheader("🔍 Content-Based Recommendation")

    # TF-IDF Vectorization
    product_descriptions = df.drop_duplicates(subset=['StockCode'])[['StockCode', 'Description']].dropna()
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_descriptions['Description'])
    product_similarity = cosine_similarity(tfidf_matrix)

    # Recommendation function
    def recommend_similar_products(stock_code, product_data, similarity_matrix, num_recommendations=5):
        product_index = product_data[product_data['StockCode'] == stock_code].index[0]
        sim_scores = list(enumerate(similarity_matrix[product_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        recommended_indices = [i[0] for i in sim_scores]
        return product_data.iloc[recommended_indices]['StockCode'].tolist()

    recommendations = recommend_similar_products(product_id, product_descriptions, product_similarity)
    st.write(f"Products similar to {product_id}:")
    st.write(recommendations)

# Streamlit App Workflow
uploaded_file = st.file_uploader("📂 Upload your Online Retail Excel file", type=['xlsx'])

if uploaded_file:
    # Load and clean data
    data = load_data(uploaded_file)
    st.success("✅ Data loaded successfully!")

    # Sidebar options
    option = st.sidebar.radio("Choose an option:", ["EDA", "Popularity-Based", "Collaborative Filtering", "Content-Based"])

    if option == "EDA":
        show_eda(data)
    elif option == "Popularity-Based":
        popularity_recommendation(data)
    elif option == "Collaborative Filtering":
        customer_id_input = st.number_input("Enter Customer ID:", min_value=0, step=1)
        if st.button("Recommend"):
            collaborative_recommendation(data, customer_id_input)
    elif option == "Content-Based":
        product_id_input = st.text_input("Enter Product (Stock Code):")
        if st.button("Recommend"):
            content_based_recommendation(data, product_id_input)
