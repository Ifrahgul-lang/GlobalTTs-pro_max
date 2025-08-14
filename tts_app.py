import streamlit as st
from gtts import gTTS
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from pydub import AudioSegment, effects
from io import BytesIO
import whisper, tempfile, os, datetime, numpy as np
import speech_recognition as sr
from pytube import YouTube
import librosa, soundfile as sf

DetectorFactory.seed = 42
AudioSegment.converter = "ffmpeg"

supported_langs = {
    "English": "en", "Urdu": "ur", "Arabic": "ar", "French": "fr",
    "German": "de", "Spanish": "es", "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja", "Russian": "ru", "Hindi": "hi", "Portuguese": "pt",
    "Italian": "it", "Korean": "ko", "Turkish": "tr", "Dutch": "nl"
}

voice_styles = {
    "neutral": {"speed": 1.0, "pitch": 0, "volume": 0},
    "happy": {"speed": 1.2, "pitch": 50, "volume": 2},
    "sad": {"speed": 0.8, "pitch": -30, "volume": -3},
    "angry": {"speed": 1.1, "pitch": 20, "volume": 6},
    "whisper": {"speed": 0.9, "pitch": -50, "volume": -10},
    "robot": {"speed": 1.0, "pitch": 0, "volume": 0, "effect": "robot"}
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_text(text, target="en"):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except:
        return text

def process_audio(audio_bytes, style_params):
    audio = AudioSegment.from_file(BytesIO(audio_bytes.getvalue()))
    if style_params["speed"] != 1.0:
        audio = audio.speedup(playback_speed=style_params["speed"])
    if style_params.get("effect") == "robot":
        samples = np.array(audio.get_array_of_samples())
        samples = librosa.effects.pitch_shift(samples.astype(float), sr=audio.frame_rate, n_steps=5)
        audio = AudioSegment(samples.astype(np.int16).tobytes(), frame_rate=audio.frame_rate,
                             sample_width=audio.sample_width, channels=audio.channels)
    elif style_params["pitch"] != 0:
        new_sample_rate = int(audio.frame_rate * (2 ** (style_params["pitch"] / 1200)))
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
        audio = audio.set_frame_rate(audio.frame_rate)
    if style_params["volume"] != 0:
        audio += style_params["volume"]
    output = BytesIO()
    audio.export(output, format="mp3")
    return output

def stt_from_audio(uploaded_file):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    result = model.transcribe(tmp_path)
    os.unlink(tmp_path)
    return result["text"]

def youtube_to_text(url):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            audio_stream.download(output_path=os.path.dirname(tmp.name), filename=os.path.basename(tmp.name))
            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp.name) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
        os.unlink(tmp.name)
        return text
    except Exception as e:
        return f"Error processing YouTube video: {str(e)}"

def main():
    st.set_page_config(page_title="GlobalTTS Pro", layout="wide")
    st.title("🎙️ GlobalTTS Pro: Advanced Text-to-Speech & Speech-to-Text")
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio("Input Mode", ["Text", "Audio Upload", "YouTube URL"])
        speaker = st.text_input("Speaker Name", "GlobalTTS")
        style = st.selectbox("Voice Style", list(voice_styles.keys()))
        lang_display = st.selectbox("Output Language", list(supported_langs.keys()))
        lang = supported_langs[lang_display]
        enhance = st.checkbox("Enhance Audio Quality")
        st.markdown("---")
        st.markdown("*Note:* For best results, keep text under 500 characters.")
    col1, col2 = st.columns([2, 1])
    with col1:
        if mode == "Text":
            user_input = st.text_area("📝 Enter Text", height=200, max_chars=1000)
        elif mode == "Audio Upload":
            file = st.file_uploader("🎤 Upload Audio (WAV/MP3)", type=["wav", "mp3"])
            user_input = stt_from_audio(file) if file else ""
            if user_input:
                st.success(f"📜 Transcription: {user_input}")
        else:
            url = st.text_input("🔗 YouTube URL")
            if url:
                with st.spinner("Processing YouTube video..."):
                    user_input = youtube_to_text(url)
                    if user_input:
                        st.success(f"📜 Transcription: {user_input}")
    with col2:
        if st.button("✨ Generate Speech", use_container_width=True):
            if not user_input.strip():
                st.warning("Please provide input text or audio.")
            else:
                with st.spinner("Processing..."):
                    try:
                        src_lang = detect_language(user_input)
                        translated = translate_text(user_input, lang)
                        tts = gTTS(translated, lang=lang)
                        audio_bytes = BytesIO()
                        tts.write_to_fp(audio_bytes)
                        processed_audio = process_audio(audio_bytes, voice_styles[style])
                        st.subheader("🔊 Output")
                        st.audio(processed_audio, format="audio/mp3")
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{speaker}_{lang}_{style}_{timestamp}.mp3"
                        st.download_button("⬇️ Download Audio", data=processed_audio.getvalue(), file_name=filename,
                                           mime="audio/mp3", use_container_width=True)
                        with st.expander("🔍 Details"):
                            st.markdown(f"*Original Text ({src_lang})*: {user_input}")
                            st.markdown(f"*Translated Text ({lang})*: {translated}")
                            st.markdown(f"*Voice Style*: {style}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
