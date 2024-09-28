import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Print TensorFlow version for debugging
st.write(f"TensorFlow version: {tf.__version__}")

# Load the saved model using TFSMLayer
try:
    model = tf.keras.layers.TFSMLayer("ai_text_detector_model_v2", call_endpoint='serving_default')
    st.success("Model loaded successfully from SavedModel directory")
    
    # Try to get information about the model's input
    st.write("Attempting to inspect model structure:")
    st.write(model.get_config())
except Exception as e:
    st.error(f"Error loading or inspecting SavedModel: {str(e)}")
    st.stop()

# Recreate the exact same vectorize_layer as used during training
max_features = 75000
sequence_length = 512

def tf_lower_and_split_punct(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# You need to adapt the layer with the same data as during training
# This is a placeholder. You should save the vocabulary during training and load it here
vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(["placeholder text"]))

def predict_ai_generated(text):
    # Try both vectorized and raw text input
    vectorized_text = vectorize_layer([text])
    st.write("Vectorized text shape:", vectorized_text.shape)
    
    try:
        # Attempt with vectorized input
        prediction_vectorized = model(vectorized_text)
        st.write("Prediction with vectorized input:", prediction_vectorized)
        return prediction_vectorized[0][0]
    except Exception as e:
        st.write(f"Error with vectorized input: {str(e)}")
        
        try:
            # Attempt with raw text input
            prediction_raw = model(tf.constant([text]))
            st.write("Prediction with raw text input:", prediction_raw)
            return prediction_raw[0][0]
        except Exception as e:
            st.error(f"Error with raw text input: {str(e)}")
            return None

st.title('AI-Generated Text Detector')

user_input = st.text_area("Enter the text you want to check:", height=200)

if st.button('Predict'):
    if user_input:
        result = predict_ai_generated(user_input)
        if result is not None:
            st.write(f"Probability of being AI-generated: {result:.2%}")
            if result > 0.5:
                st.warning("This text is likely AI-generated.")
            else:
                st.success("This text is likely human-written.")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.write("Note: This model's predictions are not 100% accurate and should be used as a guide only.")
