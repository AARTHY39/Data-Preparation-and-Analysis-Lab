import pandas as pd
import itertools

products = pd.DataFrame({
    "Transaction ID": [1, 2, 3, 4, 5],
    "Products": [
        ["Laptop", "T-shirt"],
        ["Book", "T-shirt"],
        ["Laptop", "Book"],
        ["Laptop", "Headphones", "Jeans"],
        ["T-shirt", "Jeans"]
    ],
})

support_threshold = 0.2
confidence_threshold = 0.7

def mine_pairwise_rules(df):
    total_tx = len(df)
    item_counts = {}
    pair_counts = {}

    for _, row in df.iterrows():
        items = row["Products"]

        for it in items:
            item_counts[it] = item_counts.get(it, 0) + 1

        for a, b in itertools.combinations(sorted(items), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = []

    for (a, b), c_ab in pair_counts.items():
        support = c_ab / total_tx
        conf_a_b = c_ab / item_counts[a]
        conf_b_a = c_ab / item_counts[b]

        rows.append({
            "Antecedent": a,
            "Consequent": b,
            "Support": support,
            "Confidence": conf_a_b
        })

        rows.append({
            "Antecedent": b,
            "Consequent": a,
            "Support": support,
            "Confidence": conf_b_a
        })

    all_rules = pd.DataFrame(rows)

    filtered = all_rules[
        (all_rules["Support"] >= support_threshold) &
        (all_rules["Confidence"] >= confidence_threshold)
    ].reset_index(drop=True)

    return all_rules, filtered

all_rules, filtered_rules = mine_pairwise_rules(products)

print("All Possible Association Rules:")
print(
    all_rules.to_string(
        index=False,
        formatters={
            "Support": "{:.2f}".format,
            "Confidence": "{:.2f}".format
        }
    )
)

print("\nFiltered Association Rules (Support >= 0.2, Confidence >= 0.7):")
print(
    filtered_rules.to_string(
        index=False,
        formatters={
            "Support": "{:.2f}".format,
            "Confidence": "{:.2f}".format
        }
    )
)