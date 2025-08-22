# YouTube Comments Sentiment Analysis App  

An end-to-end **Streamlit web app** that performs **sentiment analysis** on YouTube video comments.  
The app supports both **classical ML models** and a **Deep Learning approach** for comparison, and visualizes results interactively.  

---

##  Features
-  **Fetch comments** from any YouTube video using its **Video ID**.  
-  **Sentiment Analysis**:
  - Classical ML (Logistic Regression, Naive Bayes, SVM)
  - Deep Learning (LSTM)  
-  **Visualizations**:
  - Sentiment distribution pie chart
  - Word clouds for positive, negative, neutral comments
  - Model performance comparison  
-  **Streamlit UI** with light theme  
-  **Secure API Key Handling**: Users input their own YouTube API key in the UI (no keys stored in code).  

---

##  Project Structure

youtube-sentiment/
│── app.py # Main Streamlit app
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── config.yaml # App settings (max comments, default values)
│
├── data/
│ └── sample_train.csv
│
├── src/
│ ├── data/
│ │ ├── fetch_comments.py # Fetch YouTube comments via API
│ │ └── preprocess.py # Text preprocessing
│ ├── models/
│ │ ├── classical.py # Classical ML pipeline (TF-IDF + models)
│ │ ├── deep_learning.py # LSTM-based DL sentiment model
│ │ └── inference.py
│ └── utils/
│   ├── io_utils.py 
│   └── viz.py # Streamlit charts & word clouds
│
└── artifacts/ # Pre-trained models, TF-IDF vectorizers
