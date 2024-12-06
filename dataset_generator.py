import pandas as pd
import numpy as np

# Define frequency columns
frequencies = ['250Hz' , '500Hz', '1000Hz','2000Hz', '3000Hz','4000Hz', '6000Hz', '8000Hz']

weights = {
    '250Hz': 0.064997,
    '500Hz': 0.067797,
    '1000Hz': 0.073496,
    '2000Hz':0.070103,
    '3000Hz': 0.062909,
    '4000Hz': 0.058483,
    '6000Hz': 0.053565,
    '8000Hz':0.048649
}

min_sum = sum([weights[freq] * 1 for freq in frequencies])
max_sum = sum([weights[freq] * 10 for freq in frequencies])


def calculate_result(scores):
    weighted_sum = sum((10 - scores[i]) * weights[frequencies[i]] for i in range(len(frequencies)))
    result = 1+ (weighted_sum - min_sum) / (max_sum - min_sum) * 4
    return round(result,3)



def generate_severe_condition(num_samples, seed=42):
    np.random.seed(seed)
    severe_scores = []

    while len(severe_scores) < 6900:
        scores = np.random.randint(7, 11, len(frequencies))
        result = calculate_result(scores)
        if result <= 1:
            severe_scores.append(scores)

    return severe_scores



def generate_bad_condition(num_samples, seed=42):
    np.random.seed(seed)
    bad_scores = []

    while len(bad_scores) < 6969:
        scores = np.random.randint(5, 8, len(frequencies))
        result = calculate_result(scores)
        if 1 < result <= 2:
            bad_scores.append(scores)

    return bad_scores


def generate_moderate_condition(num_samples, seed=42):
    np.random.seed(seed)
    moderate_scores = []

    while len(moderate_scores) < 6991:
        scores = np.random.randint(3, 6, len(frequencies))
        result = calculate_result(scores)
        if 2 < result <= 3:
            moderate_scores.append(scores)

    return moderate_scores


def generate_good_condition(num_samples, seed=42):
    np.random.seed(seed)
    good_scores = []

    while len(good_scores) < 6996:
        scores = np.random.randint(2, 5, len(frequencies))
        result = calculate_result(scores)
        if 3 < result <= 4:
            good_scores.append(scores)

    return good_scores


def generate_excellent_condition(num_samples, seed=42):
    np.random.seed(seed)
    excellent_scores = []

    while len(excellent_scores) < 6903:
        scores = np.random.randint(1, 4, len(frequencies))
        result = calculate_result(scores)
        if result > 4:
            excellent_scores.append(scores)

    return excellent_scores


def generate_full_dataset(num_samples_per_condition, seed=69):
    np.random.seed(seed)


    severe_scores = generate_severe_condition(num_samples_per_condition, seed)
    bad_scores = generate_bad_condition(num_samples_per_condition, seed + 69)
    moderate_scores = generate_moderate_condition(num_samples_per_condition, seed + 138)
    good_scores = generate_good_condition(num_samples_per_condition, seed + 22)
    excellent_scores = generate_excellent_condition(num_samples_per_condition, seed + 65)

    def create_dataframe(scores_list, condition):
        data = {freq: [scores[i] for scores in scores_list] for i, freq in enumerate(frequencies)}
        data['result'] = [calculate_result(scores) for scores in scores_list]
        data['condition'] = [condition] * len(scores_list)
        return pd.DataFrame(data)

    df_severe = create_dataframe(severe_scores, 'Severe')
    df_bad = create_dataframe(bad_scores, 'Bad')
    df_moderate = create_dataframe(moderate_scores, 'Moderate')
    df_good = create_dataframe(good_scores, 'Good')
    df_excellent = create_dataframe(excellent_scores, 'Excellent')


    df_full = pd.concat([df_severe, df_bad, df_moderate, df_good, df_excellent])


    df_full = df_full.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_full


num_samples_per_condition = 5000
df_full_dataset = generate_full_dataset(num_samples_per_condition)

df_full_dataset.to_csv('hearing_loss_full_data2set.csv', index=False)

print("Full dataset with 5000 samples per condition generated and saved as 'hearing_loss_full_data2set.csv'")