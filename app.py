import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Print TensorFlow version for debugging
st.write(f"TensorFlow version: {tf.__version__}")

# List contents of the current directory
st.write("Contents of the current directory:")
for item in os.listdir('.'):
    st.write(item)

# Check if the model directory exists
if os.path.exists('ai_text_detector_model'):
    st.write("Contents of ai_text_detector_model directory:")
    for item in os.listdir('ai_text_detector_model'):
        st.write(item)
else:
    st.error("ai_text_detector_model directory not found")

# Load the saved model using TFSMLayer
try:
    model = tf.keras.layers.TFSMLayer("ai_text_detector_model", call_endpoint='serving_default')
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Rest of your code remains the same...
# (TextVectorization layer, prediction function, etc.)

st.title('AI-Generated Text Detector')

user_input = st.text_area("Enter the text you want to check:", height=200)

if st.button('Predict'):
    if user_input:
        try:
            result = predict_ai_generated(user_input)
            st.write(f"Probability of being AI-generated: {result:.2%}")
            if result > 0.5:
                st.warning("This text is likely AI-generated.")
            else:
                st.success("This text is likely human-written.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.write("Note: This model's predictions are not 100% accurate and should be used as a guide only.")
