import torch
import torchaudio
import torchaudio.transforms as T
import os
import matplotlib.pyplot as plt
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fft_size = 1024
input_size = (fft_size // 2 + 1) * 3


def get_min_spec(db_spec):
    bands = db_spec.shape[1]
    frames = db_spec.shape[2]
    data = [10000] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            if db_spec[0][j][i] < data[j]:
                data[j] = db_spec[0][j][i]
    return data


def get_avg_spec(db_spec):
    bands = db_spec.shape[1]
    frames = db_spec.shape[2]
    data = [0] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            data[j] += db_spec[0][j][i]
    for i in range(0, bands):
        data[i] /= frames
    return data


def get_max_spec(db_spec):
    bands = db_spec.shape[1]
    frames = db_spec.shape[2]
    data = [-1000] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            if db_spec[0][j][i] > data[j]:
                data[j] = db_spec[0][j][i]
    return data


def get_audio_data(sample="kick"):
    sample_type = 0
    match sample:
        case 'kick':
            sample_type = 0
        case 'snare':
            sample_type = 1
        case 'crash':
            sample_type = 2
        case 'hi-hat':
            sample_type = 3
        case _:
            sample_type = 0

    sample_path = os.path.join('data/' + sample + '.wav')
    waveform, sample_rate = torchaudio.load(sample_path)

    amount = waveform / sample_rate
    waveforms = [0] * amount
    for i in range(amount):
        waveforms[i] = waveform[sample_rate * i:sample_rate * (i + 1)]

    spectrogram = T.Spectrogram(n_fft=fft_size)
    data = [[0], [0], [0], [0]] * amount
    for i in range(amount):
        db_spec = librosa.power_to_db(spectrogram(waveforms[i]))
        min_spec = get_min_spec(db_spec)
        avg_spec = get_avg_spec(db_spec)
        max_spec = get_max_spec(db_spec)
        data[i] = [[min_spec], [avg_spec], [max_spec], sample_type]

    return torch.tensor(data)


def generate_datasets():
    kick_data = get_audio_data('kick')
    snare_data = get_audio_data('snare')
    data = kick_data + snare_data


if __name__ == '__main__':
    generate_datasets()
