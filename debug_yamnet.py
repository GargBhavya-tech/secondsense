"""
Validate a from-scratch (no torchaudio/librosa) mel-spectrogram implementation against the
real exported yamnet.tflite, before porting the same math to Kotlin. If this produces sane
top-class predictions on a known reference audio file, the math is right.

USAGE:
    venv\\Scripts\\python.exe debug_yamnet.py
"""
import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter

MODEL = "export_assets/tflite_models/yamnet/yamnet-tflite-float/yamnet.tflite"
WAV = "C:/Users/bhavy/.qaihm/qai-hub-models/models/yamnet/v1/speech_whistling2.wav"
LABELS_PATH = "export_assets/tflite_models/yamnet/yamnet-tflite-float/labels.txt"

SAMPLE_RATE = 16000
WIN_SAMPLES = 400   # 25ms
HOP_SAMPLES = 160   # 10ms
FFT_LEN = 512
N_MELS = 64
FMIN = 125.0
FMAX = 7500.0
LOG_OFFSET = 0.001
PATCH_FRAMES = 96   # 0.96s / 10ms hop


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels, fft_len, sample_rate, fmin, fmax):
    n_fft_bins = fft_len // 2 + 1
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_len + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_mels, n_fft_bins))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_left, f_center):
            if 0 <= k < n_fft_bins:
                fbank[m - 1, k] = (k - f_left) / max(f_center - f_left, 1)
        for k in range(f_center, f_right):
            if 0 <= k < n_fft_bins:
                fbank[m - 1, k] = (f_right - k) / max(f_right - f_center, 1)
    return fbank


def wav_to_log_mel_patches(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # simple linear resample (good enough for this validation; real capture will be 16kHz native)
        n_target = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio), n_target, endpoint=False), np.arange(len(audio)), audio)

    # center-pad like torchaudio's default (reflect padding, n_fft//2 each side)
    pad = FFT_LEN // 2
    audio_padded = np.pad(audio, (pad, pad), mode="reflect")

    window = np.hanning(WIN_SAMPLES).astype(np.float32)
    n_frames = 1 + (len(audio_padded) - FFT_LEN) // HOP_SAMPLES
    fbank = mel_filterbank(N_MELS, FFT_LEN, SAMPLE_RATE, FMIN, FMAX)

    mel_frames = []
    for i in range(n_frames):
        start = i * HOP_SAMPLES
        frame = audio_padded[start:start + WIN_SAMPLES]
        if len(frame) < WIN_SAMPLES:
            break
        windowed = frame * window
        spec = np.abs(np.fft.rfft(windowed, n=FFT_LEN))  # magnitude, not power
        mel = fbank @ spec
        log_mel = np.log(mel + LOG_OFFSET)
        mel_frames.append(log_mel)

    mel_frames = np.array(mel_frames)  # [T, 64]
    n_patches = mel_frames.shape[0] // PATCH_FRAMES
    patches = mel_frames[:n_patches * PATCH_FRAMES].reshape(n_patches, PATCH_FRAMES, N_MELS)
    return patches


def main():
    labels = [l.strip() for l in open(LABELS_PATH, encoding="utf-8")]
    patches = wav_to_log_mel_patches(WAV)
    print(f"audio -> {patches.shape[0]} patches of shape {patches.shape[1:]}")

    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    print(f"model expects input shape {ind['shape']}")

    for i, patch in enumerate(patches):
        arr = patch.reshape(1, 1, PATCH_FRAMES, N_MELS).astype(np.float32)
        interp.set_tensor(ind["index"], arr)
        interp.invoke()
        scores = interp.get_tensor(outd["index"])[0]
        top5 = np.argsort(-scores)[:5]
        print(f"\npatch {i} (t={i:.1f}s..{i+1:.1f}s):")
        for idx in top5:
            print(f"  {labels[idx]:35s} {scores[idx]:.3f}")


if __name__ == "__main__":
    main()
