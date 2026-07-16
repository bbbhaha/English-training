# Mandarin L1 English Word-Deletion Strategy

## Why the previous detector was unstable

Forced alignment always tries to place every target phone somewhere in the audio. A generated
boundary therefore proves only that the aligner found a path, not that the learner spoke the word.
Duration-only rules also confuse fast speech, reductions, and alignment errors with deletion.

## Evidence from relevant research

- SpeechOcean762 contains 5,000 English utterances from 250 Mandarin-L1 speakers with expert
  sentence, word, and phone scores. Its phone score 0 combines incorrect and missed phones, so it is
  useful for pronunciation assessment but does not by itself provide clean word-deletion labels.
  Source: https://www.isca-archive.org/interspeech_2021/zhang21x_interspeech.html
- L2-ARCTIC includes Mandarin speakers and explicit phone substitution, deletion, and addition
  annotations. The project uses its Mandarin subset as the direct deletion evidence source.
  Source: https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.html
- Cao et al. show that forced-alignment GOP cannot naturally represent deletion and insertion. Their
  alignment-free CTC method marginalizes over time boundaries and obtains a 29.02% relative gain over
  the TDNN-GOP feature baseline on SpeechOcean762. The strongest scalar variant allows substitution
  and deletion. Source: https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html
- Articulatory-feature alignment improves MDD F1 by 4.9% relative on L2-ARCTIC, supporting the use of
  independent phonological/acoustic evidence instead of trusting one forced alignment.
  Source: https://www.isca-archive.org/interspeech_2022/chen22l_interspeech.html
- Anti-phone modeling improves L2-ARCTIC MDD F1 over both an E2E baseline and GOP, but primarily
  addresses substitutions and distortions. It is therefore retained for later ordinary
  mispronunciation work, not used as the primary word-deletion signal.
  Source: https://www.isca-archive.org/interspeech_2020/yan20_interspeech.html
- Luo et al. use 926-speaker COLSEC and 100-speaker CHLOE Mandarin English data to learn
  substitution, insertion, and deletion priors. Their prior-weighted two-pass confusion network
  improves scoring accuracy from 67.3% to 84.1%, showing that Mandarin-specific priors should verify
  rather than replace acoustic evidence.
  Source: https://www.isca-archive.org/interspeech_2011/luo11_interspeech.html
- Logit-based GOP evaluation on Mandarin L2 English finds that logits separate phones better than
  overconfident posterior probabilities. This supports using the CTC full-vs-deleted log-likelihood
  margin as a model feature instead of treating one saturated probability as the decision.
  Source: https://arxiv.org/abs/2506.12067

## Implemented fusion

The detector now keeps three independent evidence streams:

1. Unprompted English ASR word-edit alignment, with ASR confidence and matching left/right context.
2. Alignment-free Wav2Vec2 CTC likelihood comparison between the full target sentence and each
   one-word-deleted alternative.
3. A speaker-independent logistic fusion model trained on Mandarin L2-ARCTIC speech with controlled
   synthetic whole-word deletions.
4. Forced-alignment duration and boundary evidence as support, never as proof on its own except for
   extreme multi-phone compression.

The system confirms `deletion` when ASR and CTC agree, or when ASR deletion and extreme duration
compression agree. A single ASR omission or a single strong CTC score becomes `possible_deletion`.
If G2P cannot produce a reliable target pronunciation, the word is not diagnosed and the frontend
shows `单词暂未收录`.

## Speaker-independent fusion training

The training split follows the repository's original speaker isolation:

- train: BWC and LXC
- development and threshold selection: NCC
- untouched test speaker: TXHC

The first version contains 981 word examples, including 64 controlled synthetic whole-word
deletions. On the held-out TXHC synthetic-deletion test, the old rules obtain precision 1.000,
recall 0.625, and F1 0.769. The learned fusion obtains precision 0.923, recall 0.750, and F1 0.828.
On 84 words from untouched complete TXHC recordings, it produces zero confirmed deletion false
alarms. These numbers are separate from the small real-deletion diagnostic subset below.

Reproduce training with:

```powershell
python scripts/train_mandarin_deletion_fusion.py `
  --phones C:/path/to/data/processed/l2_arctic/phones.csv `
  --source-project-root C:/path/to/project `
  --output-dir outputs/mandarin_deletion_training_v1 `
  --model-output models/mandarin_deletion_fusion_v1.joblib `
  --utterances-per-speaker 8 `
  --deletions-per-utterance 2
```

## Local Mandarin-L1 diagnostic validation

The reproducible script `scripts/evaluate_mandarin_l2_arctic_deletions.py` was run on all Mandarin
L2-ARCTIC utterances in the local processed corpus that contain a fully deleted word. The subset has
6 real whole-word deletions. Confirmed deletion produced TP=6, FP=0, FN=0. Counting both confirmed and
possible deletion produced TP=6, FP=0, FN=0 after fusion. This is a small diagnostic subset and must not be reported
as corpus-wide or literature-comparable accuracy.

## Real paired recordings supplied by the project user

Version 2 adds six labelled training recordings from the sentence `Life was like a box of
chocolates. You never know what you're gonna get.` Six files remain held out: five different
deletions plus one complete reading. The consistency aligner expands common spoken equivalents such
as `YOU'RE -> YOU ARE` and `GONNA -> GOING TO` while preserving the original target `word_index`.

Across all 12 paired files, the final diagnostic run confirms all 11 intended deletions with no
confirmed false deletion. The complete 14-word recording has neither confirmed nor possible
deletion alarms. This is useful real-audio regression evidence, but all files appear to contain one
speaker and one sentence, so the result is not evidence of speaker-independent 100% accuracy.

Reproduce with:

```powershell
python scripts/evaluate_manual_deletion_pairs.py `
  --manifest data/manual_deletion_pairs/manifest.csv `
  --model models/mandarin_deletion_fusion_v2.joblib `
  --output-dir outputs/manual_deletion_pairs/final_v2
```
