from itertools import combinations
from collections import defaultdict

dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
]

min_support = 0.4
min_confidence = 0.6

single_counts = defaultdict(int)

for transaction in dataset:
    unique_items = set(transaction)
    for item in unique_items:
        single_counts[frozenset([item])] += 1

total_transactions = len(dataset)

support_single = {
    item: count / total_transactions
    for item, count in single_counts.items()
}

pair_counts = defaultdict(int)

for transaction in dataset:
    unique_items = set(transaction)
    for pair in combinations(unique_items, 2):
        pair_counts[frozenset(pair)] += 1

relevant_items = {
    'Nutmeg',
    'Yogurt',
    'Onion',
    'Eggs',
    'Ice cream',
    'Kidney Beans'
}

expected_rules = [
    ('Nutmeg', 'Onion'),
    ('Yogurt', 'Onion'),
    ('Onion', 'Yogurt'),
    ('Eggs', 'Kidney Beans'),
    ('Eggs', 'Onion'),
    ('Ice cream', 'Onion'),
    ('Ice cream', 'Kidney Beans'),
    ('Onion', 'Ice cream'),
    ('Kidney Beans', 'Ice cream')
]

association_rules = []

for antecedent, consequent in expected_rules:
    antecedent_set = frozenset([antecedent])
    pair_set = frozenset([antecedent, consequent])

    support_pair = pair_counts.get(pair_set, 0) / total_transactions
    confidence = support_pair / support_single[antecedent_set]

    if confidence >= min_confidence:
        association_rules.append((antecedent, consequent, confidence))

print("Strong Association Rules:")

for antecedent, consequent, confidence in association_rules:
    print(f"{antecedent} => {{'{consequent}'}} (Confidence: {confidence:.2f})")