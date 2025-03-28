import matplotlib.pyplot as plt
# Update the dataset with the corrected size for Senator Tweets (100K) and adjust labels for size in brackets
labels_extended = [
    "Reddit_Submissions (1.2M)",
    "Reddit_Submissions High Q (800K)",
    "Reddit_Submissions Mid Q (370K)",
    "Reddit_Submissions Low Div (1.6M)",
    "Webis_Reddit (1.4M)",
    "Webis_Reddit High Q (1M)",
    "Webis_Reddit Mid Q (366K)",
    "Webis_Reddit Low Div (260K)",
    "100M_Tweets (2.5M)",
    "100M_Tweets High Q (1.2M)",
    "100M_Tweets Mid Q (1.1M)",
    "Senator_Tweets (100K)"
]

# Correct the size for Senator Tweets in the labels and other data
x_values_extended = [
    0.6657993884101101,  # Reddit: 1.2M
    0.6649172350477304,  # Reddit: High Q 800K
    0.6518291226998547,  # Reddit: Mid Q 370K
    0.5824903666446114,  # Reddit: Low Div 1.6M
    0.6936496995019082,  # Webis: 1.4M
    0.6964545137864115,  # Webis: High Q 1M
    0.6690404065997906,  # Webis: Mid Q 366K
    0.633114896655581,   # Webis: Low Div 260K
    0.6365032177314678,  # Tweets: Default 2.5M
    0.65460990933338,    # Tweets: High Q 1.2M
    0.6316537455936527,  # Tweets: Mid Q 1.1M
    0.6286969503215462   # Senator Tweets (100K)
]

y_values_extended = [
    1.622,  # Reddit: 1.2M
    1.994,  # Reddit: High Q 800K
    1.002,  # Reddit: Mid Q 370K
    1.28,   # Reddit: Low Div 1.6M
    1.717,  # Webis: 1.4M
    1.994,  # Webis: High Q 1M
    1.004,  # Webis: Mid Q 366K
    1.73,   # Webis: Low Div 260K
    1.46,  # Tweets: Default 2.5M
    1.994,  # Tweets: High Q 1.2M
    1.014,  # Tweets: Mid Q 1.1M
    1.89   # Senator Tweets (100K)
]

# Assign colors based on the dataset
colors_extended = ["blue"] * 4 + ["green"] * 4 + ["orange"] * 3 + ["purple"] * 1

# Create a single scatter plot with different colors for each dataset
plt.figure(figsize=(10, 8))

# Scatter plot
plt.scatter(x_values_extended, y_values_extended, c=colors_extended, edgecolor="black", s=100, alpha=0.7)

# # Add annotations with updated labels
# for label, x, y in zip(labels_extended, x_values_extended, y_values_extended):
#     plt.text(x, y, label, fontsize=13, ha='right' if "Low Div" in label else 'left')

# Label axes and add title
plt.xlabel("Diversity")
plt.ylabel("Quality")

# Add legend
legend_labels = ["Reddit Submissions", "Webis Reddit", "100M Tweets", "Senator Tweets"]
legend_colors = ["blue", "green", "orange", "purple"]
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=lbl)
                    for c, lbl in zip(legend_colors, legend_labels)], loc="lower right")

# Grid and show plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
