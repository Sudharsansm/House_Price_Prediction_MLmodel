# 🏡 House Price Prediction — ML Model

A Machine Learning project built using **Python**, **Pandas**, **Scikit-learn**, and **Visualization tools** to predict house prices based on multiple property features.

---

## 📌 **Project Overview**

This project focuses on predicting house prices using a dataset that contains various features such as:

* Area
* Number of bedrooms
* Air conditioning
* Guestroom availability
* Basement
* Hot water facility
* Furnishing status
* Near main road
* Preferred area
* And more…

The model uses **Linear Regression** after performing data preprocessing, feature engineering, normalization, and evaluation.

---

## 📂 **Dataset Used**

The dataset is loaded from:

```
/content/drive/MyDrive/dataset/Housing.csv
```

It contains both categorical and numerical features.
Categorical features were converted into numerical form to make them ML-compatible.

---

## 🔧 **Technologies Used**

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-Learn
* Google Colab

---

## 🧹 **Data Preprocessing Steps**

1. **Loaded dataset** using `pandas.read_csv()`
2. Displayed data using `.head()`, `.info()`, `.describe()`
3. Checked missing values using `.isnull().sum()`
4. Converted categorical columns (`yes/no`) to numeric:

   ```python
   list1 = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
   df[list1] = df[list1].replace({'yes':1, 'no':0})
   ```
5. Encoded furnishing status:

   ```python
   df['furnishingstatus'].replace({'furnished':0, 'semi-furnished':1, 'unfurnished':2})
   ```
6. Standardized numerical columns using `StandardScaler`:

   ```python
   list2 = ['price', 'area']
   df[list2] = scaler.fit_transform(df[list2])
   ```
7. Created correlation heatmap & histograms for visualization.

---

## 📊 **Exploratory Data Analysis (EDA)**

* Correlation heatmap
* Distribution plots (histograms)
* Relationship between price and features
* Data type checks and summary statistics

---

## 🤖 **Model Used**

### **Linear Regression**

Trained using:

```python
lr = LinearRegression()
lr.fit(x_train, y_train)
```

Prediction:

```python
y_predict = lr.predict(x_test)
```

Evaluation Metrics:

* **R² Score**
* **Mean Squared Error**

Final accuracy:

```python
lr_accuracy = r2_score(y_test, y_predict)*100
```

---

## 📈 **Model Accuracy**

The model achieved:

### ✅ **R² Score Accuracy: ~ YOUR_VALUE_HERE %**

(Replace with your actual accuracy output.)

---

## 📁 **Project Folder Structure**

```
House_Price_Prediction_MLmodel/
│
├── House_Price.ipynb         # Main Notebook
├── Housing.csv               # Dataset (if included)
├── README.md                 # Project Documentation
└── model.pkl                 # Saved model (optional)
```

---

## 🚀 **How to Run**

1. Clone the repository:

```bash
git clone https://github.com/Sudharsansm/House_Price_Prediction_MLmodel.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open notebook:

```bash
jupyter notebook House_Price.ipynb
```

4. Run all cells to train and evaluate the model.

---

## 🔮 **Future Improvements**

* Add more ML algorithms (Random Forest, XGBoost)
* Hyperparameter tuning
* Build a web UI using Flask / FastAPI
* Deploy using Streamlit / Hugging Face Spaces

---

## 📝 **Author**

**Sudharsan SM**
Machine Learning developer 



Just tell me!
