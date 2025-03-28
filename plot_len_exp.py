import numpy as np
import matplotlib.pyplot as plt
# plot with scatter

def plot_with_slope(X, y, xlabel, ylabel, title=None, save_dir=None):
    plt.scatter(X, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # compute regression plot
    m, b = np.polyfit(X, y, 1)
    X_ = np.unique(X)
    plt.plot(X_, m*X_ + b, color='red', linewidth=0.5, linestyle='--')

    # Calculate center point for text placement
    x_center = min(X) + 0.5 * (max(X) - min(X))  # 50% from the left (center of x range)
    y_center = m * x_center + b  # y-value on the regression line at x_center

    # Add slight offset to avoid text overlapping with the line
    y_offset = - 0.05 * (max(y) - min(y))  # 2% of y range

    # add slope to figure near the center of the line
    plt.text(x_center, y_center + y_offset, 'slope = {:.4f}'.format(m), fontsize=12)

    if title:
        plt.title(title)

    # if save_dir:
    #     # todo: make dir if not exists
    #     plt.savefig(save_dir)
    #     print("Saved to: ", save_dir)
    # else:
    plt.show()
    plt.clf()

# senators
q_l_senator_tweets = np.array([
    # Q man

    # q, l, r, rel_q
    [51.12, 171.996, 0.75, 0.741],
    [51.12, 171.996, 1.0, 0.801],
    [80.0, 230.952, 0.75, 0.874],
    [80.0, 230.952, 1.0, 0.881],
    # L man
    # short
    [55.416, 132.296, 0.75, 0.738],
    [55.416, 132.296, 1.0, 0.717],
    # long
    [76.512, 267.476, 0.75, 0.845],
    [76.512, 267.476, 1.0, 0.855],
])

# q plot
X = q_l_senator_tweets[:, 0]  # q
y = q_l_senator_tweets[:, 3]  # rel_q
save_path = f"viz_results/length_experiments/senator_tweets_q.pdf"
plot_with_slope(X, y, 'Human data quality', 'Relative Quality', 'senator tweets', save_dir=save_path)


# l plot
X = q_l_senator_tweets[:, 1]  # len
y = q_l_senator_tweets[:, 3]  # rel
save_path = f"viz_results/length_experiments/senator_tweets_l.pdf"
plot_with_slope(X, y, 'Human data length', 'Relative Quality', 'senator tweets', save_dir=save_path)


# 100M
q_l_100m_tweets = np.array([
    # q, l, r, rel_q
    # Q
    [40.0,  127.432, 1.0, 0.59],
    [40.0,  127.432, 0.75, 0.529],
    [40.0,  127.432, 0.5, 0.625],
    [40.0,  127.432, 0.25, 0.7],
    [40.0,  127.432, 1/8, 0.727],
    [40.0,  127.432, 1/16, 0.872],

    [60.0,  158.98, 1.0, 0.604],
    [60.0,  158.98, 0.75, 0.579],
    [60.0,  158.98, 0.5, 0.654],
    [60.0,  158.98, 0.25, 0.797],
    [60.0,  158.98, 1/8, 0.924],
    [60.0,  158.98, 1/16, 1.007],

    [80.0,  178.72, 1.0, 0.636],
    [80.0,  178.72, 0.75, 0.575],
    [80.0,  178.72, 0.5, 0.680],
    [80.0,  178.72, 0.25, 0.891],
    [80.0,  178.72, 1/8, 0.993],
    [80.0,  178.72, 1/16, 0.977],

    # len
    [47.12, 98.8, 1.0, 0.76],
    [47.12, 98.8, 0.75, 0.677],
    [47.12, 98.8, 0.5, 0.703],
    [47.12, 98.8, 0.25, 0.757],
    [47.12, 98.8, 1/8, 0.853],
    [47.12, 98.8, 1/16, 0.912],

    [48.8,  126.676, 1.0, 0.782],
    [48.8, 126.676, 0.75, 0.687],
    [48.8, 126.676, 0.5, 0.593],
    [48.8, 126.676, 0.25, 0.802],
    [48.8, 126.676, 1/8, 0.885],
    [48.8, 126.676, 1/16, 0.944],


    [58.4,  209.512, 1.0, 0.639],
    [58.4,  209.512, 0.75, 0.718],
    [58.4,  209.512, 0.5, 0.622],
    [58.4,  209.512, 0.25, 0.811],
    [58.4,  209.512, 1/8, 0.921],
    [58.4,  209.512, 1/16, 0.945]
])

# keep only those where ratio is <0.5
q_l_100m_tweets = q_l_100m_tweets[q_l_100m_tweets[:, 2] <= 0.5]

# q plot
X = q_l_100m_tweets[:, 0]  # q
y = q_l_100m_tweets[:, 3]  # rel_q
save_path = f"viz_results/length_experiments/100M_tweets_q.pdf"
plot_with_slope(X, y, 'Human data quality', 'Relative Quality', '100M tweets', save_dir=save_path)


# l plot
X = q_l_100m_tweets[:, 1]  # len
y = q_l_100m_tweets[:, 3]  # rel
save_path = f"viz_results/length_experiments/100M_tweets_l.pdf"
plot_with_slope(X, y, 'Human data length', 'Relative Quality', '100M tweets')