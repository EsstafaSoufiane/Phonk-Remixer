import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import librosa
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_harmonic_part(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    harmonic, _ = librosa.effects.hpss(y)
    sf.write(output_path, harmonic, sr)
    return harmonic, sr

def generate_808_bass(sr, duration, base_freq=55, slide_freq=110):
    t = np.linspace(0, duration, int(sr * duration))
    freq = np.linspace(base_freq, slide_freq, len(t))
    wave = np.sin(2 * np.pi * freq * t) * 0.5
    bass_file = os.path.join(OUTPUT_FOLDER, '808_bass.wav')
    write(bass_file, sr, (wave * 32767).astype(np.int16))
    return bass_file

def pitch_shift_vocal(vocal_path, factor=1.2):
    vocal = AudioSegment.from_wav(vocal_path)
    pitched_vocal = vocal._spawn(vocal.raw_data, overrides={
        "frame_rate": int(vocal.frame_rate * factor)
    }).set_frame_rate(vocal.frame_rate)
    pitched_vocal_file = os.path.join(OUTPUT_FOLDER, 'pitched_vocal.wav')
    pitched_vocal.export(pitched_vocal_file, format='wav')
    return pitched_vocal_file

def combine_elements(harmonic_path, bass_path, beat_path, vocal_path):
    harmonic = AudioSegment.from_wav(harmonic_path)
    bass = AudioSegment.from_wav(bass_path)
    vocal = AudioSegment.from_wav(vocal_path)
    
    total_duration = len(harmonic)
    
    kick = AudioSegment.from_wav(os.path.join(UPLOAD_FOLDER, 'kick.wav'))
    snare = AudioSegment.from_wav(os.path.join(UPLOAD_FOLDER, 'snare.wav'))
    hi_hat = AudioSegment.from_wav(os.path.join(UPLOAD_FOLDER, 'hi_hat.wav'))
    
    drum_pattern = AudioSegment.silent(duration=0)
    
    pattern_duration = 2000  
    
    pattern = AudioSegment.silent(duration=pattern_duration)
    
    pattern = pattern.overlay(kick, position=0)
    pattern = pattern.overlay(kick, position=1000)
    
    pattern = pattern.overlay(snare, position=500)
    pattern = pattern.overlay(snare, position=1500)
    
    for i in range(8):
        pattern = pattern.overlay(hi_hat, position=i * 250)
    
    while len(drum_pattern) < total_duration:
        drum_pattern += pattern
    
    drum_pattern = drum_pattern[:total_duration]
    
    harmonic = harmonic - 3  
    bass = bass - 2  
    drum_pattern = drum_pattern - 4  
    vocal = vocal - 2  
    
    combined = harmonic.overlay(bass)
    combined = combined.overlay(drum_pattern)
    combined = combined.overlay(vocal)
    
    combined_file = os.path.join(OUTPUT_FOLDER, 'phonk_remix.wav')
    combined.export(combined_file, format='wav')
    return combined_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'classical' not in request.files or 'vocal' not in request.files or \
       'kick' not in request.files or 'snare' not in request.files or 'hi_hat' not in request.files:
        return jsonify({'error': 'Missing files! Ensure you upload all required files.'}), 400

    classical_file = os.path.join(UPLOAD_FOLDER, 'classical.wav')
    vocal_file = os.path.join(UPLOAD_FOLDER, 'vocal.wav')
    kick_file = os.path.join(UPLOAD_FOLDER, 'kick.wav')
    snare_file = os.path.join(UPLOAD_FOLDER, 'snare.wav')
    hi_hat_file = os.path.join(UPLOAD_FOLDER, 'hi_hat.wav')

    request.files['classical'].save(classical_file)
    request.files['vocal'].save(vocal_file)
    request.files['kick'].save(kick_file)
    request.files['snare'].save(snare_file)
    request.files['hi_hat'].save(hi_hat_file)

    harmonic_file = os.path.join(OUTPUT_FOLDER, 'harmonic.wav')
    harmonic, sr = extract_harmonic_part(classical_file, harmonic_file)

    bass_file = generate_808_bass(sr, duration=5)
    pitched_vocal_file = pitch_shift_vocal(vocal_file)

    phonk_remix_file = combine_elements(harmonic_file, bass_file, None, pitched_vocal_file)

    return jsonify({
        'audio': f'/output/phonk_remix.wav',
        'message': 'Phonk remix successfully created!'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)