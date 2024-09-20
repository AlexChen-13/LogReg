import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title('Приложение для классификации')
st.write('Приложение позволяет загрузить датасет, провести классификацию по признакам, узнать веса признаков и нарисовать график для 2 указанных признаков')

uploaded_file = st.sidebar.file_uploader('Загружаем файл', type='csv')

tips = pd.read_csv(uploaded_file)
st.write(tips.head(5))


ss = StandardScaler()

tips.iloc[:, :-1] = ss.fit_transform(tips.iloc[:, :-1])

class LogReg:
       
    def __init__(self, learning_rate = 0.1, n_iters =1000):
        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
    def fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)

        n_inputs = X.shape[1]
        self.coef_ = np.random.uniform(-5, 5, n_inputs)
        self.intercept_ = np.random.uniform(-5, 5)
        
        
        for epoch in range(self.n_iters):

            y_pred = self.sigmoid(self.intercept_ + X@self.coef_)
            error = (y - y_pred)
            
            w0_grad = -2 * error 
            w_grad = -2 * X * error.reshape(-1, 1) 
            
            self.coef_ = self.coef_ - self.learning_rate * w_grad.mean(axis=0) 
            self.intercept_ = self.intercept_ - self.learning_rate * w0_grad.mean() 

    def predict(self, X):
        X = np.array(X) 
        y_pred =  X@self.coef_ + self.intercept_
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
        
    def predict_proba(self, X):
        X = np.array(X)
        y_pred = self.sigmoid(self.intercept_ + X @ self.coef_)
        return np.vstack((1 - y_pred, y_pred)).T
    

my_model = LogReg()
X = tips.iloc[:, :-1]
y = tips.iloc[:,-1]
my_model.fit(X, y)

coefs = np.round(my_model.coef_.tolist(), 3)
new_dict = dict(zip(X, coefs))
st.write(new_dict)

feature1 = st.selectbox("Введите название первого признака", X.columns)
feature2 = st.selectbox("Введите название второго признака", X.columns)

x1_range = np.linspace(X[feature1].min(), X[feature1].max(), 100)
x2_range = np.linspace(X[feature2].min(), X[feature2].max(), 100)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
grid = np.c_[xx1.ravel(), xx2.ravel()]

probas = my_model.predict_proba(grid)[:, 1].reshape(xx1.shape)


fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(tips[tips['Personal.Loan'] == 1][feature1], 
           tips[tips['Personal.Loan'] == 1][feature2], 
           color='red', label='Personal Loan = 1')

ax.scatter(tips[tips['Personal.Loan'] == 0][feature1], 
           tips[tips['Personal.Loan'] == 0][feature2], 
           color='blue', label='Personal Loan = 0')

contour = ax.contour(xx1, xx2, probas, levels=[0.5], linewidths=2, colors='black')
ax.clabel(contour, inline=True, fontsize=8)

ax.set_xlabel('{feature1}')
ax.set_ylabel('{feature2}')
ax.set_title('Scatter plot of {feature1} vs {feature2} colored by Personal Loan')
ax.legend()

st.pyplot(fig)