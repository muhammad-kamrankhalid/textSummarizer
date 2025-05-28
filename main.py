import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------------------------
# 💅 Custom Styles (Black Button + Background)
# ---------------------------
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: black;
        color: white;
        border: none;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🌄 Background image
set_background("https://play-lh.googleusercontent.com/1XD0j6QiiGetCi8Rh6HnaXgIUV7GjDkScb3EgvmhbMNPN1OKUHDS6Ton3pOYAZ-Aq5py")

# ---------------------------
# 🔄 Load model from current directory
# ---------------------------
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(".")
    model = T5ForConditionalGeneration.from_pretrained(".")
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# 📋 Streamlit App UI
# ---------------------------
st.title("📝 Text Summarizer using T5")
st.write("This app summarizes long text using a fine-tuned T5 model.")

user_input = st.text_area("Enter the paragraph to summarize", height=250)
max_input_length = st.slider("Max Input Length", 100, 512, 512)
max_output_length = st.slider("Max Summary Length", 20, 100, 50)

# ---------------------------
# 🧠 Generate Summary
# ---------------------------
if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_text = "summarize: " + user_input.strip()
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

        summary_ids = model.generate(
            input_ids,
            max_length=max_output_length,
            min_length=10,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("🧠 Summary")
        st.success(summary)
