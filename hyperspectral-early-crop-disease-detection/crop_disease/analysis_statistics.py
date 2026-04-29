import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------- LOAD DATA ----------------
X = np.load("data/X_cleaned.npy")   # spectral values
y = np.load("data/Y_labels.npy")    # class labels

print("Shape of spectral data:", X.shape)
print("Shape of labels:", y.shape)

# We analyze one wavelength band (example: band 50)
band = X[:, 50]

# ---------------- STATISTICS ----------------
print("\n===== STATISTICAL ANALYSIS (Band 50) =====")

mean = np.mean(band)
median = np.median(band)
mode = pd.Series(band).mode()[0]
variance = np.var(band)
std_dev = np.std(band)
minimum = np.min(band)
maximum = np.max(band)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std_dev)
print("Min:", minimum)
print("Max:", maximum)

# ---------------- HISTOGRAM ----------------
plt.figure(figsize=(6,4))
plt.hist(band, bins=30, color='green', edgecolor='black')
plt.title("Histogram of Spectral Reflectance (Band 50)")
plt.xlabel("Reflectance Value")
plt.ylabel("Frequency")
plt.savefig("results/histogram_band50.png")
plt.show()

# ---------------- BOX PLOT ----------------
plt.figure(figsize=(5,4))
sns.boxplot(x=band, color="orange")
plt.title("Boxplot of Spectral Values")
plt.savefig("results/boxplot.png")
plt.show()

# ---------------- CLASS DISTRIBUTION PIE ----------------
unique, counts = np.unique(y, return_counts=True)

labels = ['Background','Healthy','Stressed','Diseased'][:len(unique)]

plt.figure(figsize=(5,5))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.title("Class Distribution")
plt.savefig("results/class_distribution.png")
plt.show()

# ---------------- MEAN SPECTRAL SIGNATURE ----------------
mean_signature = np.mean(X, axis=0)

plt.figure(figsize=(8,4))
plt.plot(mean_signature)
plt.title("Average Spectral Signature of Crop")
plt.xlabel("Wavelength Band")
plt.ylabel("Reflectance")
plt.savefig("results/mean_signature.png")
plt.show()