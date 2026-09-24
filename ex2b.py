import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.linear_model import RANSACRegressor, LinearRegression
import matplotlib.pyplot as plt

def remove_outliers(data, threshold=2.0):
    mean = np.mean(data)
    std = np.std(data)
    filtered = [x if abs(x - mean) <= threshold * std else np.nan for x in data]
    return np.array(filtered)

def apply_median_filter(data, filter_size=3):
    return medfilt(data, kernel_size=filter_size)

def apply_moving_average(data, window_size=3):
    return pd.Series(data).rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).mean().to_numpy()

def perform_robust_regression(x, y):
    x = x.reshape(-1, 1)
    model = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=10
    )
    model.fit(x, y)
    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask
    return x[inlier_mask], y[inlier_mask], x[outlier_mask], y[outlier_mask]

def apply_kalman_filter(data, measurement_noise=1.0, process_noise=0.01):
    n = len(data)
    x_est = np.zeros(n)
    p = np.zeros(n)

    x_est[0] = data[0]
    p[0] = 1.0

    for k in range(1, n):
        x_pred = x_est[k - 1]
        p_pred = p[k - 1] + process_noise

        k_gain = p_pred / (p_pred + measurement_noise)

        x_est[k] = x_pred + k_gain * (data[k] - x_pred)
        p[k] = (1 - k_gain) * p_pred

    return x_est

np.random.seed(42)

x = np.linspace(0, 10, 100)
true_y = 3 * x + 5

noise = np.random.normal(0, 5, size=x.shape)
y = true_y + noise

y[::10] += 30

filtered_outliers = remove_outliers(y, threshold=2.0)

filtered_median = apply_median_filter(y, filter_size=5)

smoothed_avg = apply_moving_average(y, window_size=5)

inlier_x, inlier_y, outlier_x, outlier_y = perform_robust_regression(x, y)

filtered_kalman = apply_kalman_filter(
    y,
    measurement_noise=4,
    process_noise=0.5
)

print("Original Data with Noise (first 10):", y[:10])
print("Outlier Removed Data (first 10):", filtered_outliers[:10])
print("Median Filtered Data (first 10):", filtered_median[:10])
print("Moving Average Smoothed Data (first 10):", smoothed_avg[:10])
print("Kalman Filtered Data (first 10):", filtered_kalman[:10])
print(f"Robust Regression: {len(inlier_x)} inliers, {len(outlier_x)} outliers")

plt.figure(figsize=(12, 8))

plt.plot(x, y, 'k.', label='Noisy Data')
plt.plot(x, filtered_outliers, 'ro', label='Outlier Removed')
plt.plot(x, filtered_median, 'g-', label='Median Filter')
plt.plot(x, smoothed_avg, 'b-', label='Moving Average')
plt.plot(x, filtered_kalman, 'm-', label='Kalman Filter')
plt.plot(inlier_x, inlier_y, 'co', label='Robust Inliers')
plt.plot(outlier_x, outlier_y, 'yx', label='Robust Outliers')

plt.legend()
plt.title("Noise Handling  Mechanisms")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()