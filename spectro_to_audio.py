import os
import numpy as np
import librosa
import soundfile as sf

def load_spectrogram(file_path):
    """
    Carrega um espectrograma a partir de um arquivo .npy e ajusta a forma.
    Preserva múltiplas dimensões.
    """
    spectrogram = np.load(file_path)
    
    spectrogram = np.squeeze(spectrogram)
    
    print(f"Espectrograma carregado com a forma: {spectrogram.shape}")
    return spectrogram

def save_audio_from_spectrogram_with_griffin_lim(spectrogram, output_path, sr=16000, n_fft=1024, hop_length=256, n_iter=10):
    """
    Reconstrói áudio a partir de um espectrograma de magnitude usando o algoritmo Griffin-Lim
    para estimar a fase, preservando múltiplas dimensões.
    
    Parâmetros:
    - spectrogram: O espectrograma de magnitude
    - output_path: Caminho para salvar o arquivo de áudio .wav
    - sr: Taxa de amostragem (padrão: 16000 Hz)
    - n_fft: Tamanho da FFT (padrão: 1024)
    - hop_length: Comprimento do salto (padrão: 256)
    - n_iter: Número de iterações do algoritmo Griffin-Lim para estimar a fase (padrão: 50)
    """
    if spectrogram.size == 0:
        print(f"Erro: Espectrograma vazio ou malformado em {output_path}")
        return
    
    if spectrogram.ndim == 3:
        audio_segments = []
        for i in range(spectrogram.shape[2]):
            segment = spectrogram[:, :, i]
            audio_segment = librosa.griffinlim(segment, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
            audio_segments.append(audio_segment)
        
        audio = np.concatenate(audio_segments)
    else:
        audio = librosa.griffinlim(spectrogram, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
    
    sf.write(output_path, audio, sr)
    print(f"Áudio salvo em: {output_path}")

def process_spectrogram_directory(input_dir, output_dir, sr=16000, n_fft=510, hop_length=256):
    """
    Processa todos os arquivos de espectrograma em um diretório, gerando áudio para cada um
    usando o algoritmo Griffin-Lim, preservando múltiplas dimensões.
    
    Parâmetros:
    - input_dir: Diretório de entrada com os arquivos de espectrograma (.npy)
    - output_dir: Diretório de saída para salvar os arquivos de áudio (.wav)
    - sr: Taxa de amostragem (padrão: 16000 Hz)
    - n_fft: Tamanho da FFT (padrão: 1024)
    - hop_length: Comprimento do salto (padrão: 256)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(input_dir, filename)
            spectrogram = load_spectrogram(file_path)
            
            output_file = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(output_dir, output_file)
            
            save_audio_from_spectrogram_with_griffin_lim(spectrogram, output_path, sr, n_fft, hop_length)

input_directory = 'D:/POS/auto_dasr/dataset/reconstruct/spectrograms'
output_directory = 'D:/POS/auto_dasr/dataset/reconstruct/wav'
process_spectrogram_directory(input_directory, output_directory)
