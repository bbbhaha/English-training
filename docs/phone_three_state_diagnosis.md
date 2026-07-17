# Phone-level three-state diagnosis

## Goal

For every target phone in uploaded learner speech, the deployed backend emits one of:

- `correct` (`读对`)
- `mispronounced` (`读错`)
- `deleted` (`漏读`)

Words that cannot be converted to a supported target phone remain an explicit
`unavailable` exception and are displayed as `单词暂未收录`; they are not silently
treated as correct.

## Method

The primary acoustic backend is the Apache-2.0
[`mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme`](https://huggingface.co/mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme)
checkpoint. It was fine-tuned on manually annotated L2-ARCTIC speech and emits
character-level IPA labels. The independent
[`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
phone recognizer is the deployment safety reference.

For each target ARPAbet phone, `ctc_phone_diagnosis.py` compares three CTC
sequence hypotheses over the complete utterance:

1. Keep the canonical phone.
2. Delete the canonical phone.
3. Replace it with a restricted same-class or Mandarin-L1 confusion candidate.

The model also computes a maximum-logit GOP margin from a CTC Viterbi path.
These features follow the motivation of alignment-free CTC pronunciation
assessment and logit-based GOP:

- [Cao et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html)
- [Parikh et al., Interspeech 2025, phonological substitutions](https://www.isca-archive.org/interspeech_2025/parikh25_interspeech.html)
- [Parikh et al., Interspeech 2025, logit GOP](https://www.isca-archive.org/interspeech_2025/parikh25b_interspeech.html)

The deployed V5 decision learns from both acoustic models:

- Both models' margins, recognized-phone matches, and cross-model agreement are
  classifier inputs.
- A substitution is suppressed when the primary L2-ARCTIC recognizer matches
  the target or one of the narrow standard vowel variants `AA/AO`, `AH/IH`, or
  `UH/UW`.
- Connected-speech devoicing of final `/v/` in `OF` is protected when the
  target word and recognized alternative provide that exact context.
- A deletion is rejected when both recognizers match the target, both deletion
  margins are at most `3.0`, and the aligned segment is at least `70 ms`.
- The old post-classifier reference hard gates are disabled in V5. Independent
  evaluation showed that they reduced macro-F1 from `0.736` to `0.536` by
  suppressing too many real errors, while the classifier already contained the
  same reference evidence and passed the correct-audio safety constraint.
- A confirmed whole-word deletion from ASR and word CTC still overrides both
  phone models and marks every target phone in the word as `deleted`.

## Training and held-out result

The deployed classifier in `models/phone_three_state_v5.joblib` uses:

- dual-CTC deletion, substitution, target-logit, and path-probability evidence;
- agreement and disagreement features between the two phone recognizers;
- target phone, phone group, and aligned phone duration;
- class-balanced histogram gradient boosting.

Feature extraction covers all 600 Mandarin L2-ARCTIC utterances. Classifier
selection and thresholds use speaker-isolated partitions:

- initial train: `BWC`, `LXC` (300 utterances);
- threshold development: `NCC` (150 utterances);
- final refit: `BWC`, `LXC`, `NCC` (450 utterances);
- untouched test: `TXHC` (150 utterances).

Held-out TXHC results on 4,944 target phones:

| State | Precision | Recall | F1 |
|---|---:|---:|---:|
| correct | 0.942 | 0.972 | 0.957 |
| mispronounced | 0.698 | 0.530 | 0.603 |
| deleted | 0.726 | 0.602 | 0.658 |

- Accuracy: `0.919`
- Macro-F1: `0.739`
- Correct-phone false alarm rate: `0.028`

For comparison, the previous V2 result was accuracy `0.901`, macro-F1
`0.707`, and correct-phone false alarm rate `0.031`. V5 therefore improves
both error classes without increasing the held-out false-alarm rate.

The safety threshold was constrained by the supplied `correct.wav`, without
using the held-out `TXHC` speaker. Full-pipeline results on the supplied pair:

| Recording | correct | mispronounced | deleted |
|---|---:|---:|---:|
| `correct.wav` | 42 | 1 | 1 |
| `was.wav` (word `WAS` omitted) | 41 | 0 | 3 |

Thus the known-correct recording has a `2/44 = 4.55%` phone false-alarm rate,
while all three target phones in omitted `WAS` are `deleted`.

This is a speaker-held internal result on a subset selected for error coverage,
not a directly comparable full-corpus paper benchmark. The upstream checkpoint
may also have seen part of L2-ARCTIC during its own fine-tuning, so an external
SpeechOcean762 and newly recorded human review set remain necessary for a final
unbiased claim.

SpeechOcean762 was evaluated as a training-only auxiliary. Adding all 47,076
official-train phone rows reduced development macro-F1 to `0.707`; keeping only
2,564 high-confidence incorrect phones plus 5,128 matched correct phones still
reduced it to `0.718`. These candidates are retained as experiment artifacts
but are not deployed because score-derived SpeechOcean labels do not align
cleanly enough with the L2-ARCTIC perceived-phone three-state target.

## Reproduction

Download all runtime acoustic models:

```powershell
python scripts\download_deletion_models.py
```

Train the lightweight three-state classifier:

```powershell
python scripts\train_phone_three_state_model.py `
  --phones C:\path\to\data\processed\l2_arctic\phones.csv `
  --corpus-root C:\path\to\project `
  --feature-cache outputs\phone_three_state\l2_arctic_dual_ctc_features_full.csv `
  --resume-feature-extraction `
  --correct-sanity-features outputs\phone_three_state\correct_ensemble_sanity_features.csv `
  --classifier hist_gradient_boosting `
  --refit-on-train-dev `
  --output models\phone_three_state_v5.joblib `
  --report outputs\phone_three_state\training_report_v5.json
```

Evaluate a saved feature table:

```powershell
python scripts\evaluate_phone_three_state_model.py `
  --features outputs\phone_three_state\l2_arctic_dual_ctc_features_full.csv `
  --model models\phone_three_state_v5.joblib `
  --split test `
  --output outputs\phone_three_state\test_metrics.json
```

Run a new recording:

```powershell
python scripts\predict_pronunciation.py `
  --audio learner.wav `
  --text "SHE SEES THE BLUE BIRD" `
  --output outputs\demo\prediction.csv `
  --word-summary-output outputs\demo\word_summary.csv `
  --decision-mode phone_diagnosis `
  --enable-asr `
  --enable-ctc-deletion
```

The principal output columns are `phone_state`, `phone_state_zh`,
`recognized_phone`, `reference_recognized_phone`, the three
`phone_probability_*` fields,
`ctc_deletion_margin`, `ctc_substitution_margin`, `ctc_logit_margin`, and
`reference_ctc_deletion_margin`, `reference_deletion_supported`, and
`evidence_summary`.
