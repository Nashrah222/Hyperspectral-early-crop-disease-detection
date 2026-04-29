"""
Run this FIRST before anything else.
It tells you exactly what label values are in your ground truth
so you can configure everything correctly.
"""
import numpy as np

y = np.load('data/Y_labels.npy')

print("=" * 50)
print("LABEL DIAGNOSTIC REPORT")
print("=" * 50)
print(f"Total pixels:     {len(y):,}")
print(f"Unique labels:    {np.unique(y)}")
print(f"Label dtype:      {y.dtype}")
print()
print("Pixel count per label:")
for label in np.unique(y):
    count = np.sum(y == label)
    pct = count / len(y) * 100
    print(f"   Label {label:>4}:  {count:>10,} pixels  ({pct:.1f}%)")

print()
print("ACTION: Look at the label values above.")
print("  - Label with the MOST pixels = likely Background (class 0)")
print("  - Remaining labels = your plant health classes")
print("  - Copy the unique labels list and paste it into the")
print("    LABEL_CONFIG section of cnn_model.py and evaluate.py")
print("=" * 50)