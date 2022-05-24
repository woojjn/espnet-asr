import os
import warnings
import wave
import librosa
import soundfile
from pydub import AudioSegment

from asr_inference import Speech2Text


warnings.filterwarnings("ignore")

ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"


def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate

def main():
    speech2text = Speech2Text(
        asr_train_config=ASR_TRAIN_CONFIG, 
        asr_model_file=ASR_MODEL_FILE, 
        device='cuda',
        dtype='float32',
        )

    audio_path = "/opt/ml/input/chunks"
    # audio_file = "/opt/ml/input/espnet-asr/evalset/ksponspeech/wavs/KsponSpeech_E00001.wav"

    for file_name in sorted(os.listdir(audio_path)):
        print(file_name)
        audio_file = os.path.join(audio_path, file_name)
        audio, rate = downsampling(audio_file, sampling_rate=16000)
        duration = len(audio)/rate
        print(duration)

        result = speech2text(audio)
        print(result[0][0])

        with open("/opt/ml/input/AAYN32.txt", "a") as f:
            f.write(result[0][0] + "\n")

if __name__ == "__main__":
    main()