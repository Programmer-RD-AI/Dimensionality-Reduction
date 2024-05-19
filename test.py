from pprint import pprint


def apriori(transactions: dict, min_support: int) -> dict:
    # Generate initial candidate itemsets (C1)
    item_count = {}
    for transaction in transactions:
        for item in transaction:
            item = frozenset([item])  # Ensure each item is a frozenset for consistency
            if item in item_count:
                item_count[item] += 1
            else:
                item_count[item] = 1

    # Filter out items that don't meet the minimum support to form L1
    Lk = {item for item, count in item_count.items() if count >= min_support}
    k = 1
    frequent_itemsets = [
        set()
    ]  # Start with an empty set to index frequent sets by size

    # Main loop to generate Lk from Ck
    while Lk:
        frequent_itemsets.append(Lk)
        Ck_plus_1 = set()

        # Join step: Generate Ck+1 from Lk by finding all pairs of frequent item sets that can be merged
        Lk_list = list(Lk)
        for i in range(len(Lk_list)):
            for j in range(i + 1, len(Lk_list)):
                itemset1, itemset2 = Lk_list[i], Lk_list[j]
                new_candidate = itemset1.union(itemset2)
                if len(new_candidate) == k + 1:
                    Ck_plus_1.add(new_candidate)

        # Test each candidate in Ck+1 for minimum support
        candidate_count = {candidate: 0 for candidate in Ck_plus_1}
        for transaction in transactions:
            for candidate in Ck_plus_1:
                if candidate.issubset(transaction):
                    candidate_count[candidate] += 1

        # Form Lk+1 from candidates that meet the minimum support
        Lk = {
            candidate
            for candidate, count in candidate_count.items()
            if count >= min_support
        }
        k += 1

    # Return the union of all Lk
    return {item for sublist in frequent_itemsets for item in sublist}


# Example usage:
transactions = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "eggs"},
    {"milk", "diaper", "beer", "coke"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "coke"},
]

min_support = 2
frequent_itemsets = apriori(transactions, min_support)
pprint(frequent_itemsets)
