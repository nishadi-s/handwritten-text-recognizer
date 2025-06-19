# ✍️ Handwriting Paragraph OCR App

A powerful web-based OCR tool that recognizes **multi-line handwritten text** and converts it into clean, readable digital paragraphs using Microsoft's **TrOCR** model.

This application preserves line and word order for high-quality text recognition and offers an interactive interface via **Streamlit**.

---

## 🚀 Features

- 📄 Upload and recognize multi-line handwritten text
- ✨ Smart line and word segmentation for better accuracy
- 🤖 Powered by `microsoft/trocr-base-handwritten` from Hugging Face
- 💬 Converts handwriting to clean digital text
- 🖥️ Simple UI using Streamlit


---

## 🛠️ Setup

To set up and run the app locally, follow these steps:

```bash
 1. Clone the repository
git clone https://github.com/nishadi-s/handwriting-paragraph-ocr.git
cd handwriting-paragraph-ocr

 2. (Optional but recommended) Create and activate a virtual environment
python -m venv ocrenv
# On Windows
ocrenv\Scripts\activate
# On macOS/Linux
source ocrenv/bin/activate

 3. Install the required dependencies
pip install -r requirements.txt

 4. Run the Streamlit app
streamlit run app.py
