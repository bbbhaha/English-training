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

## Implemented fusion

The detector now keeps three independent evidence streams:

1. Unprompted English ASR word-edit alignment, with ASR confidence and matching left/right context.
2. Alignment-free Wav2Vec2 CTC likelihood comparison between the full target sentence and each
   one-word-deleted alternative.
3. Forced-alignment duration and boundary evidence as support, never as proof on its own except for
   extreme multi-phone compression.

The system confirms `deletion` when ASR and CTC agree, or when ASR deletion and extreme duration
compression agree. A single ASR omission or a single strong CTC score becomes `possible_deletion`.
If G2P cannot produce a reliable target pronunciation, the word is not diagnosed and the frontend
shows `单词暂未收录`.

## Local Mandarin-L1 diagnostic validation

The reproducible script `scripts/evaluate_mandarin_l2_arctic_deletions.py` was run on all Mandarin
L2-ARCTIC utterances in the local processed corpus that contain a fully deleted word. The subset has
6 real whole-word deletions. Confirmed deletion produced TP=6, FP=0, FN=0. Counting both confirmed and
possible deletion produced TP=6, FP=1, FN=0. This is a small diagnostic subset and must not be reported
as corpus-wide or literature-comparable accuracy.
