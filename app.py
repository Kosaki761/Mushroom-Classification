import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
import tensorflow as tf
import numpy as np



# Load your trained model
model = tf.keras.models.load_model('keras_model.h5')

# Define the class labels
class_labels = ['Agaricus', 'Amanita', 'Boletus','Cortinarius','Entoloma','Hygrocybe','Lactarius','Russula','Suillus']  # Replace with your own class labels

# Create a function to preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Create a function to perform image classification
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_idx = np.argmax(predictions)
    predicted_label = class_labels[predicted_idx]
    return predicted_label

def confidence(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_idx]
    return confidence



# Create the Streamlit app
def main():

    
    selected = option_menu( 
        menu_title=None,
        options=["About","Project","Shrooms"],
        icons=["None","None","None"],
        menu_icon="None",
        default_index=0,
        orientation="horizontal",
        styles = {
            
        }
    )
    
    
    if selected == "About":
       st.title(f"Poisonous Mushroom Detection System using CNN")
       st.header("About")
       st.markdown("<p style='text-align: justify;'>Mushroom poisoning is often the result of ingesting wild mushrooms after misidentifying a poisonous mushroom as an edible species. The most common reason for this misidentification is the general morphological and colour similarity of the poisonous mushrooms to the edible species. Consumption of poisonous mushrooms can lead to severe health complications, including death, depending on the type of mushroom. These mushrooms can grow in different environments, including homes and gardens, posing a significant risk to human health.</p>", unsafe_allow_html=True)
       st.header("Objective")
       st.write("1. To study Convolutional Neural Network (CNN) model and characteristics of mushroom imaging.")
       st.write("2. To develop poisonous mushroom detection system using CNN.")
       st.write("3. To evaluate the accuracy of CNN in the poisonous mushroom detection system.")
    

    
    if selected == "Project":
        st.title("Poisonous Mushroom Detection")
        st.write("Upload an image and the model will predict its class.")
        file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if file is not None:
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("")
            

            # # Classify the image and display the result
            predicted_label = classify_image(image)


            accuracy = confidence(image)
            accuracy = accuracy*100
            st.write(f"Predicted Class: {predicted_label}")
            st.write(f"Confidence: {accuracy:.2f} %")


    if selected == "Shrooms":
        st.title("Mushroom Available")

        tab1, tab2, tab3 = st.tabs(["All", "Edible", "Poisonous"])
        with tab1:
            # st.divider()
            with st.expander("Agaricus"):
                col1,col2 = st.columns(2)
                st.write("Traits : Rounded, gills, usually white or brown.")
                st.write("Sample Image : ")
                st.image("Media\Agaricus.jpg")
            # st.divider()
            with st.expander("Amanita"):
                st.write("Traits :  Variable colors, distinct ring around the stalk.")
                st.write("Sample Image : ")
                st.image("Media\Amanita.jpg")
            # st.divider()
            with st.expander("Boletus"):
                st.write("Traits : Often large, pores instead of gills, various colors.")
                st.write("Sample Image : ")
                st.image("Media\Boletus.jpg")
            # st.divider()
            with st.expander("Cortanarius"):
                st.write("Traits : Various colors, often with a partial veil on the stalk.")
                st.write("Sample Image : ")
                st.image("Media\Cortinarius.jpg")
            # st.divider()
            with st.expander("Entoloma"):
                st.write("Traits : Various colors, gills often pink or lilac.")
                st.write("Sample Image : ")
                st.image("Media\Entoloma.jpg")
            # st.divider()
            with st.expander("Hygrocybe"):
                st.write("Traits : Brightly colored, often bell-shaped.")
                st.write("Sample Image : ")
                st.image("Media\Hygrocybe.jpg")
            # st.divider()
            with st.expander("Lactarius"):
                st.write("Traits : Various colors, exudes a milky latex when injured.")
                st.write("Sample Image : ")
                st.image("Media\Lactarius.jpg")
            # st.divider()
            with st.expander("Russula"):
                st.write("Traits : Various colors, often brittle and flesh-like.")
                st.write("Sample Image : ")
                st.image("Media\Russula.jpg")
            # st.divider()
            with st.expander("Suillus"):
                st.write("Traits : Slimy or dry, often with small pores on the underside.")
                st.write("Sample Image : ")
                st.image("Media\Suillus.jpg")

        with tab2:
            # st.divider()
            with st.expander("Agaricus"):
                st.write("Traits : Rounded, gills, usually white or brown.")
                st.write("Sample Image : ")
                st.image("Media\Agaricus.jpg")
            # st.divider()
            with st.expander("Boletus"):
                st.write("Traits : Often large, pores instead of gills, various colors.")
                st.write("Sample Image : ")
                st.image("Media\Boletus.jpg")
            # st.divider()
            with st.expander("Suillus"):
                st.write("Traits : Slimy or dry, often with small pores on the underside.")
                st.write("Sample Image : ")
                st.image("Media\Suillus.jpg")

        with tab3:
            # st.divider()
            with st.expander("Amanita"):
                st.write("Traits :  Variable colors, distinct ring around the stalk.")
                st.write("Sample Image : ")
                st.image("Media\Amanita.jpg")
            with st.expander("Cortanarius"):
                st.write("Traits : Various colors, often with a partial veil on the stalk.")
                st.write("Sample Image : ")
                st.image("Media\Cortinarius.jpg")
            # st.divider()
            with st.expander("Entoloma"):
                st.write("Traits : Various colors, gills often pink or lilac.")
                st.write("Sample Image : ")
                st.image("Media\Entoloma.jpg")
            # st.divider()
            with st.expander("Hygrocybe"):
                st.write("Traits : Brightly colored, often bell-shaped.")
                st.write("Sample Image : ")
                st.image("Media\Hygrocybe.jpg")
            # st.divider()
            with st.expander("Lactarius"):
                st.write("Traits : Various colors, exudes a milky latex when injured.")
                st.write("Sample Image : ")
                st.image("Media\Lactarius.jpg")
            # st.divider()
            with st.expander("Russula"):
                st.write("Traits : Various colors, often brittle and flesh-like.")
                st.write("Sample Image : ")
                st.image("Media\Russula.jpg")
            
        
        

# Run the app
if __name__ == '__main__':
    main()
