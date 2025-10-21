import streamlit as st
import cv2
import pytesseract
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from PIL import Image
import io

warnings.filterwarnings("ignore")

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# Load and train model
@st.cache_resource
def load_and_train_model():
    import urllib.request
    import os
    
    # Try to load dataset from multiple possible locations
    dataset_files = ["archive(7).csv"]
    dataset_file = None
    
    for file in dataset_files:
        if os.path.exists(file):
            dataset_file = file
            break
    
    # If no file found, try to download
    if dataset_file is None:
        st.info("Dataset not found. Downloading... This may take a few minutes on first run.")
        url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        
        try:
            # Download and extract
            urllib.request.urlretrieve(url, "dataset.zip")
            import zipfile
            with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("dataset.zip")
            dataset_file = "training.1600000.processed.noemoticon.csv"
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            st.info("Please upload the dataset file (archive(7).csv) to your GitHub repository.")
            return None, None, None, None
    
    # Load dataset
    df = pd.read_csv(dataset_file, encoding='ISO-8859-1', header=None)
    df.columns = ['Sentiment', 'ID', 'Date', 'Query', 'User', 'Text']
    
    # Keep relevant columns
    df = df[['Sentiment', 'Text']]
    
    # Convert labels
    df['Sentiment'] = df['Sentiment'].map({0: "Negative", 4: "Positive"})
    
    # Balance dataset
    min_class_size = df['Sentiment'].value_counts().min()
    balanced_df = df.groupby('Sentiment').apply(lambda x: x.sample(min_class_size, random_state=42)).reset_index(drop=True)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df['Text'], 
        balanced_df['Sentiment'], 
        test_size=0.2, 
        random_state=42
    )
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test

# Extract text from image
def extract_text(image):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Extract text
        text = pytesseract.image_to_string(gray)
        return text.strip()
    
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Classify text sentiment
def classify_text_sentiment(text, model):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    sentiment = model.predict([text])[0]
    return sentiment

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Sentiment Analysis from Images",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Social Media Sentiment Analysis")
    st.markdown("### Extract text from images and analyze sentiment using Naive Bayes")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app extracts text from uploaded images using OCR (Tesseract) "
        "and classifies the sentiment as Positive or Negative using a "
        "Naive Bayes classifier trained on Twitter data."
    )
    
    # Load model
    with st.spinner("Loading model... This may take a moment."):
        try:
            result = load_and_train_model()
            if result[0] is None:
                st.error("Failed to load model. Please check the dataset.")
                return
            model, accuracy, X_test, y_test = result
            st.sidebar.success(f"Model loaded! Accuracy: {accuracy:.2%}")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Make sure the dataset file 'training.1600000.processed.noemoticon.csv' is in the same directory.")
            return
    
    # Tabs
    tab1, tab2 = st.tabs(["📸 Analyze Image", "📊 Model Performance"])
    
    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image containing text...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                
                # Extract text
                with st.spinner("Extracting text from image..."):
                    extracted_text = extract_text(image)
                
                if extracted_text:
                    st.text_area("Extracted Text", extracted_text, height=150)
                    
                    # Classify sentiment
                    with st.spinner("Analyzing sentiment..."):
                        sentiment = classify_text_sentiment(extracted_text, model)
                    
                    # Display result with color
                    if sentiment == "Positive":
                        st.success(f"### Sentiment: {sentiment} 😊")
                    else:
                        st.error(f"### Sentiment: {sentiment} 😞")
                else:
                    st.warning("No text detected in the image. Please try another image.")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        # Generate predictions for confusion matrix
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
            plt.title('Confusion Matrix')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
