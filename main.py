import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import librosa

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

fft_size = 2048
win_length = 1024

input_size = (fft_size // 2 + 1) * 2
hidden_size = 800
num_classes = 3
num_epochs = 10
# batch_size = 4
learning_rate = 0.01

fig, axs = plt.subplots(16, num_classes * 2)


def plot_max_frequency(db_spec, title="Max frequencies", ax=None):
    bands = db_spec.shape[0]
    frames = db_spec.shape[1]
    data = [-1000] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            if db_spec[j][i] > data[j]:
                data[j] = db_spec[j][i]
    ax.plot(range(0, bands), data, linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, bands - 1])
    ax.set_title(title)


def plot_average_frequency(db_spec, title="Average frequencies", ax=None):
    bands = db_spec.shape[0]
    frames = db_spec.shape[1]
    data = [0] * bands
    for i in range(0, frames):
        for j in range(0, bands):
            data[j] += db_spec[j][i]
    for i in range(0, bands):
        data[i] /= frames
    ax.plot(range(0, bands), data, linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, bands - 1])
    ax.set_title(title)


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


def get_audio_data(sample_name="kick", test=False):
    match sample_name:
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

    sample_path = os.path.join('data/' + sample_name + '_test' * test + '.wav')
    waveform, sample_rate = torchaudio.load(sample_path)

    sample_size = sample_rate // 4
    amount = waveform.shape[1] // sample_size

    waveforms = [0] * amount
    for i in range(amount):
        waveforms[i] = waveform[0][sample_size * i:sample_size * (i + 1)]

    spectrogram = T.Spectrogram(n_fft=fft_size, win_length=win_length)
    data = [0] * amount
    for i in range(amount):
        db_spec = librosa.power_to_db(spectrogram(waveforms[i]))
        data[i] = []

        # plot_max_frequency(db_spec, title=(sample_name + ' max'), ax=axs[i][2*sample_type])
        # plot_average_frequency(db_spec, title=(sample_name + ' avg'), ax=axs[i][2*sample_type + 1])
        # plot_spectrogram(spectrogram(waveforms[i]), title=sample_name, ax=axs[i][sample_type])

        min_spec = get_min_spec(db_spec)
        avg_spec = get_avg_spec(db_spec)
        max_spec = get_max_spec(db_spec)

        data[i] = avg_spec + max_spec

        # print('Completed sample ' + str(i + 1) + ' out of ' + str(amount) + ' from "' + sample_name + '"')

    labels = [0] * amount
    for i in range(amount):
        labels[i] = []
        for j in range(num_classes):
            label = 0 + (j == sample_type)
            labels[i].append(label)

    return torch.tensor(labels, dtype=torch.float32), torch.tensor(data, dtype=torch.float32)


class DrumDataset(Dataset):
    def __init__(self, test=False):
        print('Starting "kick"')
        kick_results, kick_data = get_audio_data('kick', test=test)
        print('Completed "kick"\n')

        print('Starting "snare"')
        snare_results, snare_data = get_audio_data('snare', test=test)
        print('Completed "snare"\n')

        print('Starting "crash"')
        crash_results, crash_data = get_audio_data('crash', test=test)
        print('Completed "crash"\n')

        # plt.show()

        data = torch.cat((kick_data, snare_data, crash_data))
        results = torch.cat((kick_results, snare_results, crash_results))

        self.n_samples = data.shape[0]
        self.x_data = data
        self.y_data = results
        print('Completed Dataset. Data shape:', self.x_data.shape, self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.sig = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.soft = nn.Softmax()

    def forward(self, x):
        out = self.l1(x)
        out = self.sig(out)
        out = self.l2(out)
        out = self.soft(out)
        return out


if __name__ == '__main__':
    dataset_train = DrumDataset()
    dataloader_train = DataLoader(dataset=dataset_train, shuffle=True)

    dataset_test = DrumDataset(test=True)
    dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)

    examples = iter(dataloader_train)
    features, labels = next(examples)

    model = NeuralNet(input_size, hidden_size, num_classes)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(dataloader_train)
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(dataloader_train):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            # print(labels)
            # print(outputs)
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if (epoch % 10) == 9: print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

    corrects = [0] * 3
    amounts = [0] * 3
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        i = 0
        for features, labels in dataloader_test:
            features = features.to(device)
            labels = labels.to(device)
            _, answer = torch.max(labels, 1)
            outputs = model(features)

            _, prediction = torch.max(outputs, 1)

            if prediction != answer:
                print(i)
                print(labels)
                print(outputs)


            n_samples += 1
            n_correct += (prediction == answer)

            amounts[answer] += 1
            corrects[answer] += (prediction == answer)

            i += 1

        acc = 100.0 * n_correct / n_samples
        print(f'\nAccuracy of the network: {acc} %\n')
        print(f'Accuracy of the kicks: {100.0 * corrects[0] / amounts[0]} %')
        print(f'Accuracy of the snares: {100.0 * corrects[1] / amounts[1]} %')
        print(f'Accuracy of the crashes: {100.0 * corrects[2] / amounts[2]} %')

