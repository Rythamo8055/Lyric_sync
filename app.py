# app.py
import base64  # Needed for the autoplay audio hack
import os
import re
import time

import streamlit as st
import torch
import whisper_timestamped as whisper
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

st.set_page_config(layout="wide")

st.title("🎤 One-Click Lyric Sync App")
st.info(
    "Upload songs to process them. Then, select a song and press 'Play & Synchronize' for immediate, synchronized playback."
)

# --- Session State Initialization ---
if "processed_songs" not in st.session_state:
    st.session_state.processed_songs = []
if "current_song_index" not in st.session_state:
    st.session_state.current_song_index = None
if "sync_started" not in st.session_state:
    st.session_state.sync_started = False
if "start_time" not in st.session_state:
    st.session_state.start_time = 0.0


# --- Helper Functions ---
def generate_lrc_content(segments):
    lrc_lines = []
    for segment in segments:
        start = segment["start"]
        minutes, seconds = divmod(start, 60)
        lrc_lines.append(
            f"[{int(minutes):02d}:{seconds:05.2f}]{segment['text'].strip()}"
        )
    return "\n".join(lrc_lines)


def parse_lrc(lrc_text):
    lyrics_data = []
    for line in lrc_text.splitlines():
        match = re.match(r"\[(\d{2}):(\d{2}\.\d{2})\](.*)", line)
        if match:
            minutes, seconds, text = match.groups()
            time_in_seconds = int(minutes) * 60 + float(seconds)
            lyrics_data.append((time_in_seconds, text))
    return lyrics_data


def get_audio_base64(file_path):
    """Reads an audio file and returns its base64 encoded version for embedding in HTML."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_current_song(index):
    st.session_state.current_song_index = index
    st.session_state.sync_started = False


# --- Sidebar for Uploading and Processing ---
st.sidebar.header("Process New Songs")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more audio files", type=["mp3", "wav"], accept_multiple_files=True
)

if uploaded_files:
    did_process_a_file = False
    processed_names = [song["name"] for song in st.session_state.processed_songs]
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in processed_names:
            did_process_a_file = True
            with st.sidebar.status(
                f"Processing {uploaded_file.name}...", expanded=True
            ) as status:
                try:
                    status.write("Saving file...")
                    temp_dir = "/tmp/streamlit_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_audio_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if uploaded_file.type == "audio/mpeg":
                        audio_info = MP3(temp_audio_path)
                    else:
                        audio_info = WAVE(temp_audio_path)
                    song_duration = audio_info.info.length

                    status.write("Transcribing with AI...")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper.load_model("medium", device=device)
                    result = whisper.transcribe(model, temp_audio_path, language="en")

                    status.write("Saving LRC data...")
                    lrc_data = generate_lrc_content(result["segments"])

                    st.session_state.processed_songs.append(
                        {
                            "name": uploaded_file.name,
                            "temp_audio_path": temp_audio_path,
                            "duration": song_duration,
                            "lrc_data": lrc_data,
                        }
                    )
                    status.update(
                        label=f"Processed {uploaded_file.name}!", state="complete"
                    )
                except Exception as e:
                    status.update(
                        label=f"Error processing {uploaded_file.name}", state="error"
                    )
                    st.sidebar.error(f"Details: {e}")

    if did_process_a_file:
        st.rerun()

# --- Main Area for Playlist and Playback ---
st.header("🎵 Your Playlist")
if not st.session_state.processed_songs:
    st.info("Upload songs using the sidebar to build your playlist.")
else:
    for i, song in enumerate(st.session_state.processed_songs):
        st.button(song["name"], on_click=set_current_song, args=(i,))

# --- Playback Controls and Display ---
if st.session_state.current_song_index is not None:
    st.divider()
    song = st.session_state.processed_songs[st.session_state.current_song_index]
    st.header(f"Now Playing: {song['name']}")

    # --- ONE-CLICK PLAY LOGIC ---
    if not st.session_state.sync_started:
        if st.button("▶️ Play & Synchronize", type="primary"):
            st.session_state.sync_started = True
            st.session_state.start_time = time.time()
            st.rerun()  # Rerun to start the autoplay and the sync loop

    progress_placeholder = st.empty()
    lyrics_placeholder = st.empty()

    if st.session_state.sync_started:
        # Autoplay the audio using embedded HTML
        audio_base64 = get_audio_base64(song["temp_audio_path"])
        audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
        st.markdown(audio_html, unsafe_allow_html=True)

        # Start the lyric sync loop
        lyrics_data = parse_lrc(song["lrc_data"])
        start_time = st.session_state.start_time
        while True:
            current_playback_time = time.time() - start_time
            progress = min(current_playback_time / song["duration"], 1.0)
            progress_placeholder.progress(
                progress,
                text=f"{int(current_playback_time // 60):02d}:{int(current_playback_time % 60):02d} / {int(song['duration'] // 60):02d}:{int(song['duration'] % 60):02d}",
            )

            current_lyric = ""
            for time_sec, text in lyrics_data:
                if current_playback_time >= time_sec:
                    current_lyric = text
                else:
                    break

            full_text_html = ""
            for _, text in lyrics_data:
                if text == current_lyric:
                    full_text_html += f"<mark><b>{text}</b></mark><br>"
                else:
                    full_text_html += f"<span style='color: #888;'>{text}</span><br>"
            lyrics_placeholder.markdown(
                f"<div style='font-size: 24px; font-weight: bold;'>{full_text_html}</div>",
                unsafe_allow_html=True,
            )

            time.sleep(0.5)
