# Online Retail Recommendation System

## Overview
The **Online Retail Recommendation System** is a data-driven project aimed at enhancing customer shopping experiences by suggesting relevant products. The system uses historical sales data to generate personalized recommendations through various methods, including popularity-based, collaborative filtering, and content-based approaches.

---

## Features
1. **Popularity-Based Recommendations**: Suggests the most frequently purchased products overall.
2. **Collaborative Filtering**: Recommends products based on the purchase behavior of similar customers.
3. **Content-Based Recommendations**: Suggests similar products using product descriptions and Natural Language Processing (NLP).
4. **Exploratory Data Analysis (EDA)**:
   - Insights into the most sold products.
   - Analysis of sales by country.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, Scipy
  - NLP: TfidfVectorizer (for content-based filtering)
- **Web Framework**: Streamlit (for interactive application)

---

## Dataset
The project utilizes a dataset containing online retail transaction records, including:
- **Invoice Number**: Unique identifier for transactions.
- **Stock Code**: Product ID.
- **Description**: Product description.
- **Quantity**: Number of items purchased.
- **Invoice Date**: Date of the transaction.
- **Unit Price**: Price per product.
- **Customer ID**: Unique identifier for customers.
- **Country**: Country where the transaction occurred.

---

## Application Workflow
### **Exploratory Data Analysis (EDA)**
- Display top-selling products and sales by country.

### **Recommendation Models**
1. **Popularity-Based Recommendations**:
   - Recommends products based on purchase frequency.
2. **Collaborative Filtering**:
   - Builds a user-item matrix to find similar customers and recommends products they purchased.
3. **Content-Based Filtering**:
   - Uses product descriptions to recommend similar items through TF-IDF vectorization.

### **Streamlit Integration**
- **Upload File**: Users upload the dataset.
- **Interactive Options**: View EDA, Popularity-Based, Collaborative, and Content-Based recommendations.

---

## How to Run the Project
1. **Install Required Libraries**:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Upload Dataset**: Use the provided `OnlineRetail.xlsx` dataset.
4. **Explore Features**: Access EDA and various recommendation methods.

---

## Project Files
- `app.py`: Streamlit application code.
- `OnlineRetail.xlsx`: Dataset for the project.

---

## Acknowledgments
- Dataset Source: Kaggle
- Libraries: Python, Streamlit, Scikit-learn, and others.

---

## Conclusion
The **Online Retail Recommendation System** effectively analyzes customer purchase data to suggest products, improving personalization in e-commerce. Its flexibility in using multiple recommendation techniques makes it versatile and powerful for enhancing user experiences.

## Made with ❤️ by Srivathsa Tirumala.
