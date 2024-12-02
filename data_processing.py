'''
pip install "deeplake<4"
pip install av
pip install deeplake[audio]
'''
import deeplake
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import string


Training_Data_Set = deeplake.load("hub://activeloop/timit-train")

sample = Training_Data_Set[1]  #access the first sample
sr = 16000 #TIMIT dataset has a 16kHz sample rate

#access the audio waveform from the 'audios' tensor
def process_timit_sample(sample):

    audio = sample['audios'].numpy() #extract the audio waveform as a NumPy array
    texts = sample['texts'].numpy() #extract the corresponding text

    if len(audio.shape) > 1:
            audio = audio.flatten()

    #plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)  # Convert to decibel scale

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    print(texts)

process_timit_sample(sample)


#Function to get Mel-spectrogram 2D array
def mel_spectrogram2d(sample):
    audio = sample['audios'].numpy()  #Extract the audio waveform as a NumPy array
    if len(audio.shape) > 1:
        audio = audio.flatten()

    #Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)  #Convert to decibel scale

    return normalize_melspectogram(mel_spec_db)

#Function to get label
def get_label(sample):
    text = sample['texts'].numpy()  #Extract the text as a NumPy array
    return text.item()  #Convert the NumPy scalar to a standard Python string

#Function to normalize melspectogram to set values between 0 and 1 db
def normalize_melspectogram(mel_spec):
    min_val = np.min(mel_spec)
    max_val = np.max(mel_spec)
    return (mel_spec - min_val) / (max_val - min_val)

#Generate Mel-spectrogram
normalized_mel_spectrogram = mel_spectrogram2d(sample)


#Plot the Mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(normalized_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Mel Frequency')
plt.show()


#Dictionary that has keys nbs from 0 to 26 and values the letters a->z + ' ' as value for key 26.   We are using the ascii values thats what char(97+i) is doing
id_to_char = {i: chr(97 + i) if i < 26 else ' ' for i in range(27)}


#Inverting the dictionary
char_to_id = {v: k for k, v in id_to_char.items()}

#Char to id function
def transcription_to_ids(transcription):
    return [char_to_id[char] for char in transcription]


#id to char function
def ids_to_transcription(transcription):
    return [id_to_char[id] for id in transcription]


#Testing both functions
print(transcription_to_ids('hello world'))

print(ids_to_transcription([7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]))

#adding padding to spectograms
def pad_spectrograms(spectrograms):
    #Find the maximum time steps in the batch
    max_time_steps = max(s.shape[1] for s in spectrograms)

    #Pad each spectrogram with zeros along the time axis
    padded_spectrograms = [
        np.pad(s, ((0, 0), (0, max_time_steps - s.shape[1])), mode="constant", constant_values=-1)
        for s in spectrograms
    ]
    return np.array(padded_spectrograms)


#Adding padding to lables
def pad_text_sequences(sequences):
    # Find the maximum sequence length
    max_length = max(len(seq) for seq in sequences)

    # Pad each sequence with the padding value
    padded_sequences = [
        seq + [-1] * (max_length - len(seq))
        for seq in sequences
    ]
    return np.array(padded_sequences)

#transforms upper case characters to lower case and removes punctuation
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


#function to get spectograms from batch
def get_spectograms(batch):
    return [mel_spectrogram2d(sample) for sample in batch]

#function to get labels in ids instead of text from batch
def get_labels_transformed_to_ids(batch):
    text_sequences=[get_label(sample) for sample in batch]
    text_sequences_without_capital_letters=[preprocess_text(text) for text in text_sequences]
    return [transcription_to_ids(sample) for sample in text_sequences_without_capital_letters]

#Testing
batch=[Training_Data_Set[0],Training_Data_Set[1],Training_Data_Set[2]]

spectograms=get_spectograms(batch)
text_sequences=get_labels_transformed_to_ids(batch)

print(spectograms)
print(text_sequences)

print(pad_spectrograms(spectograms))
print(pad_text_sequences(text_sequences))

#Adding masking
def create_spectrogram_mask(spectrograms):
    #A time step is valid if all its frequency bins are non-zero
    return (spectrograms != -1).astype(np.float32)

def create_text_mask(sequences):
    #Our output doesnt include -1 remeber out valid outputs are 0-26
    return (sequences != -1).astype(np.float32)


print(create_spectrogram_mask(pad_spectrograms(spectograms)))
print(create_text_mask(pad_text_sequences(text_sequences)))
