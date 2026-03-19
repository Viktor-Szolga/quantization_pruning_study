mask = np.ones(len(self.all_item_ids), dtype=bool)
mask[list(user_items)] = False

filtered_ids = np.array(self.all_item_ids)[mask]
filtered_prob = prob[mask]
filtered_prob /= filtered_prob.sum()

neg_items = np.random.choice(
    filtered_ids,
    size=self.num_negatives,
    replace=False,
    p=filtered_prob
)