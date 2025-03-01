import os
import time
import threading
import hashlib
import pickle
import numpy as np
import pyaudio
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from scipy.io import wavfile

NFFT = 4096
noverlap = 2048
threshold_percentile = 80
neighborhood_size = (20, 20)
fan_value = 10

fingerprint_db = {}
song_titles = {}

def fingerprint(audio, fs):
    f, t, Sxx = spectrogram(audio, fs, nperseg=NFFT, noverlap=noverlap)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    threshold = np.percentile(Sxx_log, threshold_percentile)
    local_max = maximum_filter(Sxx_log, size=neighborhood_size)
    peaks = (Sxx_log == local_max) & (Sxx_log > threshold)
    peak_indices = np.argwhere(peaks)
    if peak_indices.size > 0:
        peak_indices = peak_indices[np.argsort(peak_indices[:, 1])]
    else:
        return []
    peak_freqs = f[peak_indices[:, 0]]
    peak_times = t[peak_indices[:, 1]]
    fingerprints = []
    num_peaks = len(peak_indices)
    for i in range(num_peaks):
        for j in range(1, fan_value):
            if (i + j) < num_peaks:
                f1 = int(peak_freqs[i])
                f2 = int(peak_freqs[i + j])
                t1 = peak_times[i]
                t2 = peak_times[i + j]
                dt = int(round(t2 - t1))
                if 0 <= dt <= 10:
                    hash_val = hashlib.sha1(f"{f1}|{f2}|{dt}".encode()).hexdigest()
                    fingerprints.append((hash_val, t1))
    return fingerprints

def create_database(folder="songs"):
    global fingerprint_db, song_titles
    fingerprint_db = {}
    song_titles = {}
    song_id = 0
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' does not exist. Please create it and add WAV files.")
        return
    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(folder, filename)
            try:
                fs, audio = wavfile.read(filepath)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                fps = fingerprint(audio, fs)
                for h, offset in fps:
                    fingerprint_db.setdefault(h, []).append((song_id, offset))
                song_titles[song_id] = filename
                print(f"Fingerprinted: {filename} ({len(fps)} hashes)")
                song_id += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def match_fingerprint(query_hashes):
    matches = {}
    for h, query_time in query_hashes:
        if h in fingerprint_db:
            for song_id, song_time in fingerprint_db[h]:
                diff = int(round(song_time - query_time))
                matches.setdefault(song_id, {}).setdefault(diff, 0)
                matches[song_id][diff] += 1
    best_song = None
    best_count = 0
    for song_id, offsets in matches.items():
        count = max(offsets.values())
        if count > best_count:
            best_count = count
            best_song = song_id
    return best_song, best_count

def record_audio(duration=10, update_callback=None):
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)
    frames = []
    print("Recording audio from mic...")
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        if update_callback is not None:
            update_callback(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    pa.terminate()
    audio_data = b"".join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    return RATE, audio_np

def evaluate_score(score):
    if score >= 20:
        return "😃 High confidence"
    elif score >= 10:
        return "🙂 Moderate confidence"
    else:
        return "😞 Low confidence"

def update_progress(progress_var, progress_step, status_var, status_text):
    progress_var.set(progress_var.get() + progress_step)
    status_var.set(status_text)
    root.update_idletasks()

def save_fingerprint_database():
    with open("fingerprint_db.pickle", "wb") as f:
        pickle.dump(fingerprint_db, f)
    with open("song_titles.pickle", "wb") as f:
        pickle.dump(song_titles, f)
    print("Fingerprint database saved to disk.")

def load_fingerprint_database():
    global fingerprint_db, song_titles
    if os.path.exists("fingerprint_db.pickle") and os.path.exists("song_titles.pickle"):
        with open("fingerprint_db.pickle", "rb") as f:
            fingerprint_db = pickle.load(f)
        with open("song_titles.pickle", "rb") as f:
            song_titles = pickle.load(f)
        print("Loaded fingerprint database from disk.")
    else:
        print("No saved fingerprint database found, creating new one...")
        create_database("songs")
        save_fingerprint_database()

def update_visualizers(chunk_data):
    samples = np.frombuffer(chunk_data, dtype=np.int16)
    def draw_waveform():
        canvas_width = waveform_canvas.winfo_width() or 650
        canvas_height = waveform_canvas.winfo_height() or 150
        if np.max(np.abs(samples)) != 0:
            samples_norm = samples / np.max(np.abs(samples))
        else:
            samples_norm = samples
        samples_y = (canvas_height/2) - (samples_norm * (canvas_height/2))
        step = max(1, len(samples_y) // canvas_width)
        points = []
        for i in range(0, len(samples_y), step):
            x = (i / len(samples_y)) * canvas_width
            y = samples_y[i]
            points.extend([x, y])
        waveform_canvas.delete("all")
        if points:
            waveform_canvas.create_line(points, fill="lime", smooth=True)
    root.after(0, draw_waveform)

    def update_volume():
        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
        volume_percent = min(100, (rms / 32767) * 100)
        volume_progress["value"] = volume_percent
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -100
        volume_label.config(text=f"Volume: {db:.1f} dB")
    root.after(0, update_volume)

    def draw_frequency():
        freq_canvas.delete("all")
        canvas_width = freq_canvas.winfo_width() or 650
        canvas_height = freq_canvas.winfo_height() or 150
        spectrum = np.abs(np.fft.rfft(samples))
        if spectrum.size == 0:
            return
        max_val = np.max(spectrum)
        if max_val == 0:
            norm_spec = spectrum
        else:
            norm_spec = spectrum / max_val
        step = max(1, len(norm_spec) // canvas_width)
        for i in range(0, len(norm_spec), step):
            x = (i / len(norm_spec)) * canvas_width
            bar_height = norm_spec[i] * canvas_height
            freq_canvas.create_line(x, canvas_height, x, canvas_height - bar_height, fill="cyan")
    root.after(0, draw_frequency)

def start_matching_mic():
    result_frame.pack_forget()  # Hide the result frame
    control_frame.pack_forget()  # Hide the main controls
    duration_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Show duration selector
    duration_frame.lift()  # Bring to front
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Please select recording duration and click 'Start Recording'")
    result_text.config(state=tk.DISABLED)
    result_title.config(text="Select Recording Duration")

def confirm_duration():
    duration = duration_var.get()
    duration_frame.place_forget()
    proceed_with_recording(duration)

def proceed_with_recording(duration):
    result_frame.pack_forget()
    progress_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    visualizer_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,10))
    progress_var.set(0)
    status_var.set("Recording audio from microphone...")
    
    def process():
        update_progress(progress_var, 20, status_var, "Recording audio from microphone...")
        fs, audio = record_audio(duration=duration, update_callback=update_visualizers)
        update_progress(progress_var, 20, status_var, "Processing fingerprint...")
        query_hashes = fingerprint(audio, fs)
        update_progress(progress_var, 30, status_var, f"Extracted {len(query_hashes)} fingerprint hashes. Matching...")
        best_song, score = match_fingerprint(query_hashes)
        update_progress(progress_var, 30, status_var, "Match complete!")
        visualizer_frame.pack_forget()
        result_text.config(state=tk.NORMAL)
        result_text.delete("1.0", tk.END)
        if best_song is not None:
            confidence = evaluate_score(score)
            song_name = song_titles.get(best_song, 'Unknown')
            result_title.config(text=f"Identified Song: {song_name}")
            result_text.insert(tk.END, f"Match Score: {score}\nConfidence: {confidence}\n")
        else:
            result_title.config(text="No Match Found")
            result_text.insert(tk.END, "Sorry, we couldn't identify this song.\nTry again with a clearer audio sample.\n")
        result_text.insert(tk.END, "\n--- Technical Info ---\n")
        result_text.insert(tk.END, f"Sample Rate: {fs} Hz\n")
        result_text.insert(tk.END, f"Total Samples: {len(audio)}\n")
        result_text.insert(tk.END, f"NFFT: {NFFT}\n")
        result_text.insert(tk.END, f"noverlap: {noverlap}\n")
        result_text.insert(tk.END, f"Threshold Percentile: {threshold_percentile}\n")
        result_text.insert(tk.END, f"Neighborhood Size: {neighborhood_size}\n")
        result_text.insert(tk.END, f"Fan Value: {fan_value}\n")
        result_text.insert(tk.END, f"Number of fingerprint hashes extracted: {len(query_hashes)}\n")
        result_text.config(state=tk.DISABLED)
        progress_frame.pack_forget()
        control_frame.pack(pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        result_title.config(text="Ready to Identify Music")
    
    threading.Thread(target=process).start()

def start_matching_file():
    filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return
    result_frame.pack_forget()
    progress_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    progress_var.set(0)
    status_var.set(f"Processing file: {os.path.basename(filepath)}")
    
    def process():
        update_progress(progress_var, 20, status_var, f"Reading file: {os.path.basename(filepath)}")
        try:
            fs, audio = wavfile.read(filepath)
            if audio.ndim > 1:
                audio = audio[:, 0]
        except Exception as e:
            status_var.set(f"Error reading file: {e}")
            progress_var.set(100)
            root.after(2000, lambda: progress_frame.pack_forget())
            return
        update_progress(progress_var, 30, status_var, "Processing fingerprint...")
        query_hashes = fingerprint(audio, fs)
        update_progress(progress_var, 30, status_var, f"Extracted {len(query_hashes)} fingerprint hashes. Matching...")
        best_song, score = match_fingerprint(query_hashes)
        update_progress(progress_var, 20, status_var, "Match complete!")
        result_text.config(state=tk.NORMAL)
        result_text.delete("1.0", tk.END)
        if best_song is not None:
            confidence = evaluate_score(score)
            song_name = song_titles.get(best_song, 'Unknown')
            result_title.config(text=f"Identified Song: {song_name}")
            result_text.insert(tk.END, f"Match Score: {score}\nConfidence: {confidence}\n")
        else:
            result_title.config(text="No Match Found")
            result_text.insert(tk.END, "Sorry, we couldn't identify this song.\nTry again with a clearer audio sample.\n")
        result_text.insert(tk.END, "\n--- Technical Info ---\n")
        result_text.insert(tk.END, f"Sample Rate: {fs} Hz\n")
        result_text.insert(tk.END, f"Total Samples: {len(audio)}\n")
        result_text.insert(tk.END, f"NFFT: {NFFT}\n")
        result_text.insert(tk.END, f"noverlap: {noverlap}\n")
        result_text.insert(tk.END, f"Threshold Percentile: {threshold_percentile}\n")
        result_text.insert(tk.END, f"Neighborhood Size: {neighborhood_size}\n")
        result_text.insert(tk.END, f"Fan Value: {fan_value}\n")
        result_text.insert(tk.END, f"Number of fingerprint hashes extracted: {len(query_hashes)}\n")
        result_text.config(state=tk.DISABLED)
        progress_frame.pack_forget()
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    threading.Thread(target=process).start()

def refresh_database():
    progress_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    progress_var.set(0)
    status_var.set("Refreshing song database...")
    
    def process():
        update_progress(progress_var, 50, status_var, "Scanning 'songs' folder...")
        create_database("songs")
        save_fingerprint_database()
        update_progress(progress_var, 50, status_var, "Database refresh complete!")
        root.after(1500, lambda: progress_frame.pack_forget())
    
    threading.Thread(target=process).start()

load_fingerprint_database()
print("Fingerprint database ready.")

root = tk.Tk()
root.title("Pyzam - Music Recognition")
root.geometry("700x800")
root.configure(bg="#f5f5f5")

style = ttk.Style()
style.configure("TFrame", background="#f5f5f5")
style.configure("TButton", font=("Segoe UI", 12), padding=6)
style.configure("TLabel", background="#f5f5f5", font=("Segoe UI", 12))
style.configure("TProgressbar", thickness=10)

header_frame = tk.Frame(root, bg="#3498db", padx=20, pady=15)
header_frame.pack(fill=tk.X)
app_logo = tk.Label(header_frame, text="🎵", font=("Segoe UI", 28), bg="#3498db", fg="white")
app_logo.pack(side=tk.LEFT, padx=(0, 10))
app_title = tk.Label(header_frame, text="Pyzam", font=("Segoe UI", 28, "bold"), bg="#3498db", fg="white")
app_title.pack(side=tk.LEFT)
app_subtitle = tk.Label(header_frame, text="Music Recognition", font=("Segoe UI", 14), bg="#3498db", fg="white")
app_subtitle.pack(side=tk.LEFT, padx=(10, 0))

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

control_frame = ttk.Frame(main_frame)
control_frame.pack(pady=10)

button_frame = ttk.Frame(control_frame)
button_frame.pack()

mic_icon = "🎤"
file_icon = "📂"
refresh_icon = "🔄"

mic_button = tk.Button(
    button_frame, 
    text=f"{mic_icon} Listen with Mic", 
    command=start_matching_mic,
    font=("Segoe UI", 12),
    padx=15, pady=10,
    bg="#3498db", fg="white",
    relief=tk.FLAT,
    width=18
)
mic_button.grid(row=0, column=0, padx=10, pady=5)

file_button = tk.Button(
    button_frame, 
    text=f"{file_icon} Select File", 
    command=start_matching_file,
    font=("Segoe UI", 12),
    padx=15, pady=10,
    bg="#2ecc71", fg="white",
    relief=tk.FLAT,
    width=18
)
file_button.grid(row=0, column=1, padx=10, pady=5)

refresh_button = tk.Button(
    control_frame, 
    text=f"{refresh_icon} Refresh Database", 
    command=refresh_database,
    font=("Segoe UI", 10),
    bg="#f5f5f5", fg="#555",
    relief=tk.FLAT,
    borderwidth=1
)
refresh_button.pack(pady=(10, 0))

duration_frame = ttk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
duration_var = tk.IntVar(value=10)
ttk.Label(duration_frame, text="Recording Duration (seconds):", font=("Segoe UI", 12)).pack(pady=5)
ttk.Spinbox(duration_frame, from_=1, to=60, textvariable=duration_var, width=5, font=("Segoe UI", 12)).pack(pady=5)
ttk.Button(duration_frame, text="Start Recording", command=confirm_duration, style="TButton").pack(pady=10)
duration_frame.place_forget()

visualizer_frame = ttk.Frame(main_frame)

waveform_canvas = tk.Canvas(visualizer_frame, width=650, height=150, bg="black")
waveform_canvas.pack(fill=tk.BOTH, expand=True, pady=(0,5))

volume_frame = ttk.Frame(visualizer_frame)
volume_frame.pack(fill=tk.X, pady=(0,5))
volume_label = ttk.Label(volume_frame, text="Volume: - dB", font=("Segoe UI", 12))
volume_label.pack(side=tk.LEFT, padx=(0,5))
volume_progress = ttk.Progressbar(volume_frame, orient="horizontal", length=500, mode="determinate")
volume_progress.pack(side=tk.LEFT, padx=(0,5))

freq_canvas = tk.Canvas(visualizer_frame, width=650, height=150, bg="black")
freq_canvas.pack(fill=tk.BOTH, expand=True)

progress_frame = ttk.Frame(main_frame)
progress_var = tk.DoubleVar()
status_var = tk.StringVar()
status_var.set("Ready")
status_label = ttk.Label(progress_frame, textvariable=status_var)
status_label.pack(pady=(0, 5), anchor=tk.W)
progress = ttk.Progressbar(progress_frame, variable=progress_var, length=650, mode="determinate")
progress.pack(fill=tk.X, pady=5)

result_frame = ttk.Frame(main_frame)
result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
result_header = ttk.Frame(result_frame)
result_header.pack(fill=tk.X, pady=(0, 10))
result_icon_label = tk.Label(result_header, text="🎵", font=("Segoe UI", 48), fg="#3498db", bg="#f5f5f5")
result_icon_label.pack(side=tk.LEFT, padx=(0, 15))
result_title_frame = ttk.Frame(result_header)
result_title_frame.pack(side=tk.LEFT)
result_title = ttk.Label(result_title_frame, text="Ready to Identify Music", font=("Segoe UI", 16, "bold"))
result_title.pack(anchor=tk.W)
result_subtitle = ttk.Label(result_title_frame, text="Click one of the buttons above to start", font=("Segoe UI", 12))
result_subtitle.pack(anchor=tk.W)
result_text_frame = ttk.Frame(result_frame)
result_text_frame.pack(fill=tk.BOTH, expand=True)
result_text = scrolledtext.ScrolledText(result_text_frame, width=70, height=10, font=("Segoe UI", 12))
result_text.pack(fill=tk.BOTH, expand=True)
result_text.insert(tk.END, "Welcome to Pyzam!\n\nTo identify music:\n1. Click 'Listen with Mic' to record from your microphone\n   a. Select recording duration when prompted\n2. Click 'Select File' to compare from a WAV file")
result_text.config(state=tk.DISABLED)

footer_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=5)
footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
footer_text = tk.Label(footer_frame, text="Pyzam Music Recognition © 2025", font=("Segoe UI", 8), bg="#f0f0f0", fg="#888")
footer_text.pack(side=tk.RIGHT)

progress_frame.pack_forget()
root.mainloop()
