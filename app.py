import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="📝 Handwriting OCR (Paragraph Mode)", layout="centered")
st.title("✍ Enhanced Paragraph Recognition for Handwritten Text")
st.markdown("Upload a **multi-line handwritten** image. This version uses line + word order preservation and better decoding.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Cleaning and recognizing text..."):
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Preprocess
        blur = cv2.medianBlur(gray, 3)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        # Connect text lines
        line_kernel = np.ones((2, 50), np.uint8)
        dilated_lines = cv2.dilate(thresh, line_kernel, iterations=1)
        line_contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_boxes = [cv2.boundingRect(c) for c in line_contours if cv2.boundingRect(c)[3] > 15]
        line_boxes = sorted(line_boxes, key=lambda b: b[1])  # top to bottom

        paragraph = ""
        for lx, ly, lw, lh in line_boxes:
            line_img = img[ly:ly+lh, lx:lx+lw]
            line_gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
            _, line_thresh = cv2.threshold(line_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Word segmentation
            word_kernel = np.ones((1, 15), np.uint8)
            word_dilated = cv2.dilate(line_thresh, word_kernel, iterations=1)
            word_contours, _ = cv2.findContours(word_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            word_boxes = [cv2.boundingRect(c) for c in word_contours if cv2.boundingRect(c)[2] > 10]
            word_boxes = sorted(word_boxes, key=lambda b: b[0])

            line_text = ""
            for wx, wy, ww, wh in word_boxes:
                word_img = line_img[wy:wy+wh, wx:wx+ww]
                word_pil = Image.fromarray(word_img).convert("RGB")
                pixel_values = processor(images=word_pil, return_tensors="pt").pixel_values

                generated_ids = model.generate(
                    pixel_values,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
                word_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                line_text += word_text + " "

            paragraph += line_text.strip() + "\n"

    # Remove unwanted numeric tokens like '0', '1', '5f', '01'
    cleaned_paragraph = re.sub(r'\b(?:\d+[a-zA-Z]*|[a-zA-Z]*\d+)\b', '', paragraph)
    cleaned_paragraph = re.sub(r'\s{2,}', ' ', cleaned_paragraph).strip()

    st.subheader("📝 Recognized Paragraph")
    st.text_area("Extracted Text", cleaned_paragraph, height=300)
