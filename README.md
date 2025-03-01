# Pyzam - Music Recognition

Pyzam is a music recognition application that allows users to identify songs from an audio recording or file. It uses audio fingerprinting to match songs to a local database of WAV files. This application provides real-time visualizations of audio recordings and identifies music tracks with a confidence score.

## Features

- **Mic-based recognition**: Record audio from a microphone and identify music in real-time.
- **File-based recognition**: Upload a WAV file for song identification.
- **Audio fingerprinting**: Uses advanced fingerprinting algorithms to create unique identifiers for audio tracks.
- **Progress feedback**: Displays the progress of the recognition process with a progress bar and status updates.
- **Visualization**: Real-time waveform and frequency visualizations during recording.
- **Database management**: Allows for easy creation, refreshing, and saving of a local fingerprint database.

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `scipy`
  - `pyaudio`
  - `tkinter`
  - `pickle`
  - `hashlib`
  
You can install the necessary dependencies using `pip`:

```bash
pip install *dependency*
```

Note: `tkinter` is usually included with Python by default. If not, install it based on your operating system's instructions.

## Usage

### Step 1: Set up your database

Before running the application, you'll need a folder named `songs/` containing WAV files. These files will be used to create a fingerprint database for song recognition.

To create and save the fingerprint database, you can run the following command:

```bash
python pyzam.py
```

This will scan the `songs/` folder, generate fingerprints for the files, and store them in `fingerprint_db.pickle` and `song_titles.pickle`.

### Step 2: Start the Application

1. Run the application:
   ```bash
   python pyzam.py
   ```

2. The application will open a GUI with options to:
   - **Listen with Mic**: Record audio from your microphone and try to identify the song.
   - **Select File**: Choose a WAV file from your computer for song identification.
   - **Refresh Database**: Refresh the database by reprocessing the songs in the `songs/` folder.

3. Select one of the options and let the app identify the song. The result will show the song name, match score, and confidence level.

### Step 3: Manage the Database

- **Refresh Database**: Click the "Refresh Database" button to regenerate the fingerprint database from the files in the `songs/` folder.
- **Save/Load Database**: The app automatically saves and loads the fingerprint database to and from disk for future use.

## Technical Information

- **NFFT**: 4096 (Window size for FFT)
- **no overlap**: 2048 (Overlap between FFT windows)
- **Threshold percentile**: 80% (Percentile used to determine peaks in the spectrogram)
- **Neighborhood size**: (20, 20) (Size of the region for local maximum filtering)
- **Fan value**: 10 (Number of neighboring peaks considered when generating fingerprints)

## Contributing

Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!

1. Fork the repo
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the contributors of the libraries used in this project, including `numpy`, `scipy`, and `pyaudio`.
- Inspired by various music recognition tools and audio processing algorithms, Shazam in particular.
