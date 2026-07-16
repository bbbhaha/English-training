from faster_whisper import WhisperModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def main() -> None:
    WhisperModel("base.en", device="cpu", compute_type="int8")
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    print("Deletion models are ready in the local model cache.")


if __name__ == "__main__":
    main()
