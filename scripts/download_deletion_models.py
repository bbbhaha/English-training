from faster_whisper import WhisperModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


PHONE_CTC_MODEL = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
REFERENCE_PHONE_CTC_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main() -> None:
    WhisperModel("base.en", device="cpu", compute_type="int8")
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    Wav2Vec2Processor.from_pretrained(PHONE_CTC_MODEL)
    Wav2Vec2ForCTC.from_pretrained(PHONE_CTC_MODEL)
    Wav2Vec2Processor.from_pretrained(REFERENCE_PHONE_CTC_MODEL, do_phonemize=False)
    Wav2Vec2ForCTC.from_pretrained(REFERENCE_PHONE_CTC_MODEL)
    print("Word-deletion and three-state phone-diagnosis models are ready in the local model cache.")


if __name__ == "__main__":
    main()
