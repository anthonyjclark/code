from io import BytesIO
import requests
import streamlit as st
from fastai.vision.all import *

"""
# Cat Classifier

This is a simple classifier. It takes in images and
tells you if the image is of a cat or not. It was trained on
an animal dataset.
"""


def predict(img):
    st.image(img, caption="Your image", use_column_width=True)
    pred, _, probs = learn_inf.predict(img)

    f"""
    ## This **{'is' if pred == 'True' else 'is not'}** a picture of a cat.
    ### Probability of cat: {probs[1].item()*100: .2f}%
    """


def is_cat(x):
    return x[0].isupper()


path = untar_data(URLs.PETS) / "images"
learn_inf = load_learner(path / "pet_model.pkl")

# img_path = Path("trapp-hoodie.jpg")

option = st.radio("", ["Upload Image", "Image URL"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Please upload an image.")

    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        predict(img)

else:
    url = st.text_input("Please input a url.")

    if url != "":
        try:
            response = requests.get(url)
            pil_img = PILImage.create(BytesIO(response.content))
            predict(pil_img)

        except:
            st.text("Problem reading image from", url)
