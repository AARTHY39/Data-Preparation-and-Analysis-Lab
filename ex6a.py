def haar_wavelet_transform(data):
    result = []
    data_copy = data[:]

    while len(data_copy) >= 2:
        average = (data_copy[0] + data_copy[1]) / 2
        difference = (data_copy[0] - data_copy[1]) / 2

        result.append(average)
        result.append(difference)

        data_copy = data_copy[2:]

    if data_copy:
        result.extend(data_copy)

    return result


def inverse_haar_wavelet_transform(transformed_data):
    n = len(transformed_data)
    original_data = []
    i = 0

    while i < n - 1:
        average = transformed_data[i]
        difference = transformed_data[i + 1]

        original_data.append(average + difference)
        original_data.append(average - difference)

        i += 2

    if i < n:
        original_data.append(transformed_data[-1])

    return original_data


data = [5, 10, 3, 8, -2, 6, 1, 4]

transformed_data = haar_wavelet_transform(data)
print("Transformed Data:", transformed_data)

original_data = inverse_haar_wavelet_transform(transformed_data)
print("Original Data:", original_data)