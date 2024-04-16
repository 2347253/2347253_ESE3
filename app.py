
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from itertools import combinations

def load_data():
    df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
    return df

def plot_3d(df):
    fig = px.scatter_3d(df, x='Age', y='Rating', z='Positive Feedback Count', color='Rating', opacity=0.7)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig)

def plot_2d(df, x_column, y_column):
    fig = px.scatter(df, x=x_column, y=y_column, color='Rating', opacity=0.7)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig)

def image_processing():

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.subheader("Original Image")
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption="Original Image", use_column_width=True, width=200)

        st.subheader("Image Processing Options")
        processing_option = st.selectbox("Select Processing Option", ["None", "Resize", "Grayscale Conversion", "Image Cropping", "Image Rotation"])

        processed_image = original_image
        if processing_option == "Resize":
            new_size = st.slider("New Size", 10, 500, 200)
            processed_image = processed_image.resize((new_size, new_size))

        elif processing_option == "Grayscale Conversion":
            processed_image = processed_image.convert("L")

        elif processing_option == "Image Cropping":
            left = st.slider("Select Left Coordinate", 0, original_image.size[0], 0)
            top = st.slider("Select Top Coordinate", 0, original_image.size[1], 0)
            right = st.slider("Select Right Coordinate", left, original_image.size[0], original_image.size[0])
            bottom = st.slider("Select Bottom Coordinate", top, original_image.size[1], original_image.size[1])
            processed_image = processed_image.crop((left, top, right, bottom))

        elif processing_option == "Image Rotation":
            rotation_angle = st.slider("Rotation Angle", -180, 180, 0)
            processed_image = processed_image.rotate(rotation_angle)

        if processing_option != "None":
            st.subheader("Processed Image")
            st.image(processed_image, caption="Processed Image", use_column_width=True, width=200)

def text_processing(text):
    if pd.isna(text):  
        return ""  

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0  


def main():
    st.title("👗 Women’s Clothing E-Commerce Data Analysis👗")

    menu = ["Plot Visualization", "Image Processing", "Text Similarity Analysis"]
    choice = st.sidebar.selectbox("Select Option", menu)

    df = load_data()

    if choice == "Plot Visualization":
        st.subheader("3D Plot Visualization")
        plot_3d(df)

        st.subheader("2D Plot Visualization")
        x_column = st.selectbox("Select X Column", ['Age', 'Rating', 'Positive Feedback Count'])
        y_column = st.selectbox("Select Y Column", ['Rating', 'Positive Feedback Count'])
        plot_2d(df, x_column, y_column)

    elif choice == "Image Processing":
        st.subheader("Image Processing")
        image_processing()

    elif choice == "Text Similarity Analysis":
        st.subheader("Text Similarity Analysis")
        df['Processed Review'] = df['Review Text'].apply(text_processing)
        st.write(df[['Review Text', 'Processed Review']])


if __name__ == "__main__":
    main()



