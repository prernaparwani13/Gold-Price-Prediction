import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from PIL import Image
import matplotlib.pyplot as plt

# Custom CSS for futuristic look
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f0f23, #1a1a2e);
    color: #ffffff;
    font-family: 'Arial', sans-serif;
}
.stApp {
    background: transparent;
}
h1, h2, h3 {
    color: #00d4ff;
    text-shadow: 0 0 10px #00d4ff;
}
.sidebar .sidebar-content {
    background: rgba(0, 0, 0, 0.8);
    border-radius: 10px;
    padding: 20px;
}
.stButton>button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-weight: bold;
}
.stSlider {
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)

try:
    # Load data
    gold_data = pd.read_csv('gld_price_data.csv')
except FileNotFoundError:
    st.error("🚨 Data file missing. Please upload 'gld_price_data.csv'.")
    st.stop()

try:
    # Split into X and Y
    X = gold_data.drop(['Date', 'GLD'], axis=1)
    Y = gold_data['GLD']
except KeyError as e:
    st.error(f"🚨 Data column error: {e}")
    st.stop()

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

reg = RandomForestRegressor()
reg.fit(X_train, Y_train)
pred = reg.predict(X_test)
score = r2_score(Y_test, pred)

# Title with icon
st.title('🌟 Gold Price Prediction AI')

# Sidebar for inputs
with st.sidebar:
    st.header('🔮 Predict GLD Price')
    spx = st.slider('SPX Index', float(X['SPX'].min()), float(X['SPX'].max()), float(X['SPX'].mean()), step=0.01)
    uso = st.slider('USO Oil', float(X['USO'].min()), float(X['USO'].max()), float(X['USO'].mean()), step=0.01)
    slv = st.slider('SLV Silver', float(X['SLV'].min()), float(X['SLV'].max()), float(X['SLV'].mean()), step=0.01)
    eur_usd = st.slider('EUR/USD Rate', float(X['EUR/USD'].min()), float(X['EUR/USD'].max()), float(X['EUR/USD'].mean()), step=0.01)

    # Prediction
    input_data = pd.DataFrame([[spx, uso, slv, eur_usd]], columns=X.columns)
    predicted_price = reg.predict(input_data)[0]
    st.subheader('💰 Predicted GLD Price')
    st.write(f"${predicted_price:.2f}")

# Main content - simplified
col1, col2 = st.columns([1, 2])

with col1:
    try:
        img = Image.open('goldnew.jpg')
        st.image(img, width=150, caption='Gold Nugget')
    except FileNotFoundError:
        st.write('🪙 Gold Image')

    st.metric('Model Accuracy', f"{score:.2%}")

with col2:
    st.subheader('📊 Quick Insights')
    st.write(f"Data Points: {len(gold_data)}")
    st.write(f"GLD Range: ${Y.min():.2f} - ${Y.max():.2f}")

    # Simplified plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(Y_test, pred, color='#00d4ff', alpha=0.6, edgecolors='white')
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='#ff6b6b', linestyle='--', linewidth=2)
    ax.set_facecolor('#0f0f23')
    ax.tick_params(colors='white')
    ax.set_xlabel('Actual Price', color='white')
    ax.set_ylabel('Predicted Price', color='white')
    ax.set_title('Prediction Scatter', color='#00d4ff')
    st.pyplot(fig)



