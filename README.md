# Multi-scale Spectrotemporal Pattern (MSTP) for stuttering clips

An experimental, **unvalidated** multi-label method inspired by the multiscale pattern + statistical feature + feature-selection workflow in the supplied Fadile MATLAB package. This is an original adaptation to speech; it does **not** reuse the cardiac Citric Acid descriptor or claim improved accuracy.

## Proposed method

1. Resample mono audio to 16 kHz and compute an 80-band log-mel spectrogram (25 ms frames, 10 ms hop).
2. At three temporal offsets (4, 8, 16 frames), compare each center value to four temporal neighbors and separately to four frequency neighbors. Ternary states use thresholds of 0.5 times the **per-clip** standard deviation of the corresponding differences. Histograms of the 3^4 codes produce 486 pattern features (3 scales × 2 directions × 81 bins). Per-clip computations have no learned dataset statistics.
3. A log-mel summary (mean, standard deviation, 10th and 90th percentile per mel band) gives 320 statistical features. Compare stats-only, patterns-only, and concatenated features on identical splits.
4. Fit imputation, scaling, univariate feature selection (up to 256 features), and one-vs-rest logistic regression on training clips only. Pick feature count and representation by development macro average precision; tune binary thresholds on development only. Evaluate the chosen model once on the held-out test set.

Multi-label order: `Prolongation,Block,SoundRep,WordRep,Interjection`. These five categories can co-occur. This package does not convert them to mutually exclusive classes. AP measures ranking; per-label F1 is measured at development-chosen thresholds. This is a candidate method, not a demonstrated literature contribution. Establish novelty and external validity with prior-work review and independent testing.

## Input

Place audio locally; **do not commit audio or personal metadata**. Create `data/manifest.csv` from `manifest.example.csv` with one row per clip. `audio_path` is relative to the manifest directory or absolute; `group_id` must represent a real disjoint speaker identity when a speaker-independent claim is intended. An episode/show ID only establishes episode/show independence. If speaker IDs cannot be verified, label the result accordingly. Counts must be integers from 0 to 3 (three annotators); the script binarizes `count>0`. Review your dataset's count convention first. `split` must be train/dev/test and every `group_id` must occur in exactly one split. Do not use the test set while selecting methods. Dataset splits should be fixed before feature selection. Clip identity and audio hashes are audited.

## Run on the stronger computer

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py --manifest data/manifest.csv --output results/run1
```

First prepare the manifest and audio paths. The command creates `audit.json`, `features.npz`, `dev_comparison.csv`, `model.joblib`, `test_metrics.json`, `test_predictions.csv`, and `run_config.json`. Re-running with the same output path refuses to overwrite the prior run. No data, results or model are in this repository. CPU execution is supported. This is a research prototype; the full experiment requires your actual audio and manifest. Install packages and run in a fresh environment on the target machine.

### SEP-28k-Extended preparation

If using the published [SEP-28k-Extended metadata](https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended), extract clips using the [original dataset script](https://github.com/apple-aiml-research/ml-stuttering-events-dataset/blob/main/extract_clips.py). Then generate the manifest locally:

```bash
python prepare_sep28k.py --metadata /path/to/SEP-28k-Extended_clips.csv --clips /path/to/clips --output data/manifest.csv --split-column SEP28k-E --group-by speaker
```

The helper expects Apple's extracted path `clips/Show/EpId/Show_EpId_ClipId.wav`. It refuses missing audio and speaker IDs crossing train/dev/test splits. If the chosen published split conflicts with speaker identities, the helper stops; investigate the metadata instead of silently changing the split. `--group-by episode` or `--group-by show` are explicit alternatives and support only the corresponding separation claims. `--allow-missing` permits incomplete local audio, reports omitted clips, and changes the evaluated sample, so do not compare its scores directly to complete-data baselines. **Do not add the generated manifest to GitHub**, as it contains local paths and metadata.

## Design safeguards and limitations

- The held-out test is used once, after selection; ablations are compared on dev. A subsequent model change requires another untouched test set for an unbiased final estimate.
- The same source recording cut into several clips must share a group; speaker identity is preferable where available. Audio duplicate hashes across splits are rejected.
- The feature selection is learned per label from train only; label prevalence and rare categories can make AP unstable. Inspect per-label support and repeated independent group splits before drawing conclusions.
- SEP-28k style metadata often provides show/episode rather than confirmed speaker identities. Do not describe episode separation as patient/speaker separation.
- This prototype has no automatic downloader and makes no assertion about licensing or preprocessing parity with published baselines.
- The `SEP28k-E` column must be checked against your metadata version and task definition. The source dataset describes three annotator counts and five co-occurring stuttering events; the extended release provides published split columns. Attribution: [original SEP-28k](https://github.com/apple-aiml-research/ml-stuttering-events-dataset), [SEP-28k-Extended](https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended).
