import streamlit as st
import whisper
import tempfile
import os
from deep_translator import GoogleTranslator

# Function to format time for SRT
def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# Function to generate SRT content from Whisper segments
def generate_srt(segments, target_lang=None):
    srt_content = ""
    translator = GoogleTranslator(source='auto', target=target_lang) if target_lang else None

    for i, segment in enumerate(segments, start=1):
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        text = segment['text'].strip()
        
        # Translate if a target language is selected
        if translator:
            try:
                text = translator.translate(text)
            except Exception as e:
                st.warning(f"Translation failed for a segment: {e}")
                
        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content

@st.cache_resource
def load_whisper_model():
    # Load a smaller model by default for speed. 
    # Options: tiny, base, small, medium, large
    return whisper.load_model("base")

st.title("MP3 to SRT Converter")
st.write("Upload an MP3 (or audio file) to transcribe it and generate an SRT subtitle file.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

# Translation options
target_language = st.selectbox(
    "Target Subtitle Language",
    options=["Original (No Translation)", "English", "Malay"]
)

# Map UI selection to deep-translator language codes
lang_code_map = {
    "English": "en",
    "Malay": "ms",
    "Original (No Translation)": None
}
selected_lang_code = lang_code_map[target_language]

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Transcribe and Convert to SRT"):
        with st.spinner("Loading AI model... (This might log some warnings if no GPU is found, that is normal)"):
            model = load_whisper_model()
            
        with st.spinner("Transcribing audio... This may take some time depending on file length."):
            # Extract the original file extension to help FFmpeg decode it properly
            original_ext = os.path.splitext(uploaded_file.name)[1]
            
            # Whisper requires a file path, so we save the uploaded file temporarily
            # We use .getvalue() instead of .read() to avoid file pointer issues after st.audio
            buffer_data = uploaded_file.getvalue()
            if not buffer_data:
                st.error("The uploaded file is empty. Please upload a valid audio file.")
                st.stop()
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
                tmp_file.write(buffer_data)
                tmp_path = tmp_file.name
                
            try:
                # Transcribe the audio. Whisper automatically detects the language (e.g., Japanese or English)
                result = model.transcribe(tmp_path)
                
                # Generate SRT content (with optional translation)
                srt_data = generate_srt(result["segments"], target_lang=selected_lang_code)
                
                st.success("Transcription complete!")
                
                st.subheader("Generated SRT")
                st.text_area("SRT Output", srt_data, height=300)
                
                # Provide a download button for the SRT file
                st.download_button(
                    label="Download SRT File",
                    data=srt_data,
                    file_name=uploaded_file.name.rsplit('.', 1)[0] + ".srt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
                st.info("Make sure you have 'ffmpeg' installed on your system. It is required by Whisper.")
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
