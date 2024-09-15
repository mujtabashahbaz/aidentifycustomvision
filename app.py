import streamlit as st
import requests
from PIL import Image
import io

# Azure Custom Vision API details
API_KEY = 'f1941114bac543859fba9ea1749f1979'
ENDPOINT = 'https://southcentralus.api.cognitive.microsoft.com/'
PROJECT_ID = '6743fefc-c597-43a0-8d1e-4d85db9024bf'
ITERATION_ID = 'Iteration1'  # Updated Iteration ID
PREDICTION_URL = f'{ENDPOINT}customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_ID}/image'

# Function to send the image to Azure Custom Vision and return predictions
def get_prediction(image_bytes):
    headers = {
        'Prediction-Key': API_KEY,
        'Content-Type': 'application/octet-stream'
    }
    response = requests.post(PREDICTION_URL, headers=headers, data=image_bytes)
    return response.json()

# Function to display predictions in a clean, user-friendly way
def display_predictions(predictions):
    st.subheader("Prediction Results")
    
    # Sort predictions by probability in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
    
    # Display the top 5 predictions
    for i, prediction in enumerate(sorted_predictions[:5], 1):
        tag_name = prediction['tagName']
        probability = prediction['probability']
        progress_bar_color = 'green' if probability > 0.7 else 'orange' if probability > 0.3 else 'red'
        
        st.markdown(f"**Prediction {i}: {tag_name}**")
        st.progress(probability)  # Display probability as progress bar
        st.write(f"Confidence: **{probability * 100:.2f}%**")

st.title("Medical Image Classification")
st.write("Upload an image to classify using Azure Custom Vision.")

# Upload an image for classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying the image...")
    
    # Convert the uploaded image to bytes
    image = Image.open(uploaded_file)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    # Get the prediction results from Azure Custom Vision
    result = get_prediction(image_bytes)
    
    # Check if there are predictions in the API response
    if 'predictions' in result:
        predictions = result['predictions']
        if predictions:
            display_predictions(predictions)
        else:
            st.error("No predictions were returned by the model.")
    else:
        st.error("Failed to retrieve predictions. Please try again.")
else:
    st.info("Please upload an image file (jpg, jpeg, or png).")

# Add some extra information about the tool
st.sidebar.title("About")
st.sidebar.info("""
This app uses Azure Custom Vision to classify medical images. 
Simply upload an image and the AI model will attempt to classify it based on training data.
""")
