# Training dataset

Store landmark feature vectors here, one folder per gesture label.

## Layout

```
training_dataset/
  hello/
    hello_001.npy
    hello_002.npy
  yes/
    yes_001.npy
  ...
```

Each `.npy` file is a 63-dimensional float vector (21 MediaPipe hand landmarks × xyz),
normalized relative to the wrist.

## Collect real samples

```bash
python -m ml_model.collect_samples --gesture hello --count 50
```

Repeat for each gesture you want to support.

## Bootstrap with synthetic data

```bash
python -m ml_model.generate_synthetic --train
```

## Retrain after adding gestures

1. Add a new folder under `training_dataset/<gesture_name>/`
2. Add translations in `utils/language_maps.py`
3. Run:

```bash
python -m ml_model.train
```
