import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fft_size = 1024
input_size = (fft_size // 2 + 1) * 3


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def get_min_spec(db_spec):
    bands = db_spec.shape[0]
    frames = db_spec.shape[1]
    data = [10000] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            if db_spec[j][i] < data[j]:
                data[j] = db_spec[j][i]
    return data


def get_avg_spec(db_spec):
    bands = db_spec.shape[0]
    frames = db_spec.shape[1]
    data = [0] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            data[j] += db_spec[j][i]
    for i in range(0, bands):
        data[i] /= frames
    return data


def get_max_spec(db_spec):
    bands = db_spec.shape[0]
    frames = db_spec.shape[1]
    data = [-1000] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            if db_spec[j][i] > data[j]:
                data[j] = db_spec[j][i]
    return data


def get_audio_data(sample="kick"):
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

    amount = waveform.shape[1] // sample_rate
    waveforms = [0] * amount
    for i in range(amount):
        waveforms[i] = waveform[0][sample_rate * i:sample_rate * (i + 1)]

    spectrogram = T.Spectrogram(n_fft=fft_size)
    data = [0] * amount
    for i in range(amount):
        db_spec = librosa.power_to_db(spectrogram(waveforms[i]))
        data[i] = []

        min_spec = get_min_spec(db_spec)
        avg_spec = get_avg_spec(db_spec)
        max_spec = get_max_spec(db_spec)

        data[i].append(min_spec)
        data[i].append(avg_spec)
        data[i].append(max_spec)

        print('Completed sample ' + str(i+1) + ' out of ' + str(amount) + ' from "' + sample + '"')

    return (torch.zeros(amount, 1) + sample_type), torch.tensor(data)


class DrumDataset(Dataset):
    def __init__(self):
        print('Starting "kick"')
        kick_results, kick_data = get_audio_data('kick')
        print('Completed "kick"\n')

        print('Starting "snare"')
        snare_results, snare_data = get_audio_data('snare')
        print('Completed "snare"\n')

        data = torch.cat((kick_data, snare_data))
        results = torch.cat((kick_results, snare_results))

        self.n_samples = data.shape[0]
        self.x_data = data
        self.y_data = results
        print('Completed Dataset. Data shape:', self.x_data.shape, self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    dataset = DrumDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    #print(features, labels)
