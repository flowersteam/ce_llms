import time
from contextlib import contextmanager
import matplotlib.pyplot as plt

import numpy as np

from datasets import concatenate_datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from scipy.spatial import ConvexHull
from scipy.stats import sem
import imageio
import seaborn as sns
from dataset_utils import load_twitter_dataset


@contextmanager
def timer_block():
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


def plot_boxes_and_save(
        ys,
        ylabel=None, xlabel=None, fontsize=10, title=None,
        save_path=None, no_show=True, assert_n_datapoints=None
):
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.figure(figsize=(15, 10))

    labels = list(ys.keys())
    values = list(ys.values())

    # Plot
    plt.boxplot(values, labels=labels, vert=True, patch_artist=True)

    for i, val in enumerate(values):
        if assert_n_datapoints:
            assert len(val) == assert_n_datapoints, f"{len(val)} points found (expected {assert_n_datapoints}) for {labels[i]}."

        # Scatter dots slightly spread horizontally around their x-position (jitter effect)
        x_coords = [i + 1] * len(val)

        plt.scatter(x_coords, val, alpha=0.7, color='blue', edgecolor='black')

    plt.yticks(fontsize=fontsize)

    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)

    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path + ".png", dpi=300)
        # plt.savefig(save_path+".svg")
        print(f"Saved to: {save_path}")

    if not no_show:
        plt.show()
    else:
        plt.close()


def plot_and_save(
        ys,
        labels,
        ylabel=None, xlabel=None, yticks=None,
        violin=False, linewidths=None, linestyles=None, colors=None, fontsize=10, log=False,
        save_path=None, no_show=True,
        assert_n_datapoints=None, scatter=False,
):
    plt.style.use('seaborn-v0_8-darkgrid')

    assert len(labels) == len(ys)
    plt.figure(figsize=(15, 10))

    # power analysis
    def cohend(d1, d2):
        n1, n2 = len(d1), len(d2)
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        u1, u2 = np.mean(d1), np.mean(d2)
        return (u1 - u2) / s

    from statsmodels.stats.power import TTestIndPower
    def compute_N(y_a, y_b):
        effect = cohend(y_a, y_b)

        # perform power analysis
        analysis = TTestIndPower()
        result = analysis.solve_power(min(np.abs(effect), 5), power=0.8, nobs1=None, ratio=1.0, alpha=0.05/6)
        return effect, result

    # xs_ = list(ys[0].keys())
    # x_0 = 0.0
    # for l_i, lab in enumerate(labels):
    #     print(f"Lab: {lab}")
    #     for x_ in xs_:
    #         if x_ == x_0:  continue
    #         y_a = ys[l_i][x_0]
    #         y_b = ys[l_i][x_]
    #         assert len(y_a) == len(y_b)
    #         effect, result = compute_N(y_a, y_b)
    #
    #         print(f"X: {x_} effect: {effect} -> N: {result}")

    for y, label, col, lw, ls in zip(ys, labels, colors, linewidths, linestyles):

        xs = sorted(y.keys())  # e.g. generations / ratios

        if violin:
            # per text
            vp = plt.violinplot([y[g] for g in xs], showmeans=True)

            # Manually set the color
            for body in vp['bodies']:
                body.set_facecolor(col)
                # body.set_edgecolor(col)
                body.set_alpha(0.2)  # Optionally set the transparency
                # body.set_alpha(1)

            vp['cmeans'].set_color(col)
            vp['cbars'].set_color(col)
            vp['cmaxes'].set_color(col)
            vp['cmins'].set_color(col)

            ax = plt.gca()
            ax.plot([], [], c=col, label=label)

            ax.set_xticks(range(1, len(xs) + 1))
            ax.set_xticklabels(xs)

        else:
            # y = (generation -> n_seeds)
            # (n_generations)
            mean_ys = np.array([np.mean(y[x]) for x in xs])
            sems_ys = np.array([sem(y[x]) for x in xs])

            def smooth(data, n=1):
                if n > 1:
                    print("Smoothing with n = ", n)
                smooth_data = np.array([np.mean(data[max(i+1-n, 0): i+1]) for i in range(len(data))])
                return smooth_data

            mean_ys = smooth(mean_ys)
            sems_ys = smooth(sems_ys)

            if assert_n_datapoints:
                assert set(map(len, y.values())) == {assert_n_datapoints}, f"{set(map(len, y.values()))} is not all {assert_n_datapoints} for {label}"

            # col = str(lw/(max(linewidths)+5))
            plt.plot(xs, mean_ys, label=label, c=col, linewidth=lw, linestyle=ls)
            plt.fill_between(xs, mean_ys-sems_ys, mean_ys+sems_ys, alpha=0.2, color=col)

            if scatter:
                for i, x in enumerate(xs):
                    plt.scatter([x] * len(y[x]), y[x], color=col, alpha=0.6, s=5)  # Adjust `s` for point size


    if yticks:
        plt.yticks(yticks)

    if log:
        plt.gca().set_yscale('log')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)

    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)

    plt.legend(fontsize=fontsize, loc="upper left")

    if save_path:
        plt.savefig(save_path+".png", dpi=300)
        print(f"Saved to: {save_path}")
    
    if not no_show:
        plt.show()
    else:
        plt.close()




#####################################################################
## TO PLOT TOXICITY AND POLITICAL BIAS EVOLUTION ON THE SAME PLOT ###
#####################################################################

def plot_2D(y1s, y2s, labels, save_path, no_show, xlabel=None, ylabel=None):
   
    for y1, y2, label in zip(y1s, y2s, labels):
        plt.plot([np.mean(y1[i]) for i in range(len(y1))], [np.mean(y2[i]) for i in range(len(y2))], label=label, marker='o')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.hlines(0, -1, 1, linestyles='dashed', colors='black')
        plt.vlines(0, -1, 1, linestyles='dashed', colors='black')

        ## Add generation numbers
        for i in range(len(y1)):
            plt.annotate(i, (np.mean(y1[i]), np.mean(y2[i]) + 0.02))


    

    if save_path:
        plt.savefig(save_path+"_2D.png", dpi=300)
        plt.savefig(save_path+"_2D.svg")
        print(f"Saved to: {save_path}_2D")
    
    if not no_show:
        plt.show()
    else:
        plt.close()



#####################################################################
## TO PLOT THE EMBEDDINGS IN 2D WITH CONVEX HULLS AND MEAN POINTS ###
#####################################################################
def plot_repr(embs, dataset_lens, labels, save_path, gif=True, hexbin=True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'lavender', 'turquoise', 'darkorange', 'tan', 'salmon', 'gold', 'lightcoral', 'lightgreen', 'lightblue', 'lightpink', 'lightgrey', 'lightyellow', 'lightcyan', 'lightseagreen', 'lightsalmon', 'lightgoldenrodyellow']
    
    assert len(embs) == sum(dataset_lens)

    start = 0
    previous_mean = None
    image_files = []

    x_min, x_max = embs[:, 0].min(), embs[:, 0].max()
    y_min, y_max = embs[:, 1].min(), embs[:, 1].max()

    if hexbin:
        # Determine global bin size and color scale limits
        hexbin_data = []
        for d_i, d_l in enumerate(dataset_lens):
            s, e = start, start + d_l
            hexbin_data.append(embs[s:e])
            start = e

            hexbin_counts = plt.hexbin(embs[:, 0], embs[:, 1], gridsize=150, cmap='viridis', alpha=0.5).get_array()
            plt.clf()  # Clear the figure after computing global hexbin counts
            if d_i == 0:
                vmin, vmax = hexbin_counts.min(), hexbin_counts.max()
            else:
                vmin = min(vmin, hexbin_counts.min())
                vmax = max(vmax, hexbin_counts.max())
    # vmin = 0
    # vmax = 10
    extent = [x_min, x_max, y_min, y_max]
    print(f"Extent: {extent}")
    print(f"Vmin: {vmin}, Vmax: {vmax}")

    start = 0
    for d_i, d_l in enumerate(dataset_lens):
        print(f"Plotting {labels[d_i]}")
        s, e = start, start + d_l
        print(f"Start: {s}, End: {e}")
        print(len(embs[s:e]))

        if hexbin:
            hb = plt.hexbin(embs[s:e, 0], embs[s:e, 1], gridsize=150, cmap='viridis', alpha=0.5, vmin=vmin, vmax=vmax, extent=extent)
            plt.colorbar(hb, label='Count')
        else:
            plt.scatter(embs[s:e, 0], embs[s:e, 1], s=1, alpha=0.2, c=colors[d_i], label=labels[d_i])
        
        # Display mean
        plt.scatter(embs[s:e, 0].mean(), embs[s:e, 1].mean(), s=100, alpha=1, c=colors[d_i], marker='x')
        
        if previous_mean is not None:
            plt.plot([previous_mean[0], embs[s:e, 0].mean()], [previous_mean[1], embs[s:e, 1].mean()], c='black')
        
        previous_mean = embs[s:e, :].mean(axis=0)
        start = e
        points_hull = embs[s:e]
        hull = ConvexHull(points_hull)
        for simplex in hull.simplices:
            plt.plot(points_hull[simplex, 0], points_hull[simplex, 1], alpha=1, c=colors[d_i], linewidth=2, linestyle='dashed')
    
        # Add generation label
        if gif: 
            plt.title(labels[d_i])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        # plt.text(0.05, 0.95, f'Generation {d_i}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        if gif:
            # Save the current figure as an image file for GIF
            img_path = f"{save_path}_{d_i}.png"
            plt.savefig(img_path, dpi=300)
            image_files.append(img_path)
            
            plt.clf()  # Clear the figure for the next plot

    plt.legend()
    
    if gif:
        images = [imageio.imread(img_file) for img_file in image_files]
        imageio.mimsave(f"{save_path}.gif", images, duration=1000)
        print(f"Saved gif to: {save_path}.gif")

        # Clean up image files
        for img_file in image_files:
            os.remove(img_file)
    else:
        for ext in ["svg", "png"]:
            plt.title('')
            fig_save_path = f"{save_path}.{ext}"
            plt.savefig(fig_save_path, dpi=300)
            print(f"Saved to: {fig_save_path}")
        
        plt.close()

def visualize_datasets(datasets, dataset_labels, experiment_tag, gif = True, hexbin = True, embeddings_collumn="embeddings"):
    # Visualize embeddings
    dataset = concatenate_datasets(datasets, axis=0, info=None)

    dataset_lens = [len(d) for d in datasets]

    X = np.array(dataset[embeddings_collumn])
    ss_X = StandardScaler().fit_transform(X)

    # PCA
    os.makedirs("viz_results", exist_ok=True)
    print("PCA fitting")
    with timer_block():
        pca_dataset = PCA(n_components=2).fit_transform(ss_X)
        plot_repr(pca_dataset, dataset_lens, dataset_labels, save_path=f"viz_results/{experiment_tag}_pca", hexbin=hexbin, gif=gif)

    print("UMAP fitting")
    with timer_block():
        umap_X = umap.UMAP().fit_transform(ss_X)
        plot_repr(umap_X, dataset_lens, dataset_labels, save_path=f"viz_results/{experiment_tag}_umap", hexbin=hexbin, gif=gif)

    print("TSNE fiting")
    with timer_block():
        tsne_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
        plot_repr(tsne_embedded, dataset_lens, dataset_labels, save_path=f"viz_results/{experiment_tag}_tsne", hexbin=hexbin, gif=gif)





####################################################
## TO PLOT THE EVOLUTION OF METRIC DISTRIBUTIONS ###
####################################################

def plot_metric_distributions(metric_data, metric_name, saving_name, dataset = 'twitter', no_show = False, w=6, h=4, legend_i=0, legend_pos=(0.7,1), ylim=None, with_suptitle=False, fig=None, axs=None):
        # if measure != 'similarity':
            
        #     initial = self.all_initial_cumuls[measure]
        #     final = self.all_final_cumuls[measure]
        

        if metric_name == "positivity":
            clip = (-1, 1)
            linewidth = 1.5
        elif metric_name == "toxicity":
            clip = (-0.1, 1)
            linewidth = 1.5
        elif metric_name == "political_bias":
            clip = (-1, 1)
            linewidth = 1.5
        else:
            clip = None
            linewidth = 1.5

        n_generations = len(metric_data)

        
        ridge_data = []

       
        for gen in range(len(metric_data)):

            # print(f"Generation {gen}")
            #print(metric_data[gen])
            # print(np.array(metric_data[gen]).shape)

            ridge_data.append(pd.DataFrame({
                "Value": [metric_data[gen]],
                "Generation": [gen]
                
            }))
        
        
        ridge_data = pd.concat(ridge_data)
        #Convert to float
        

        g = sns.FacetGrid(ridge_data, row="Generation", aspect=15, height=0.5, sharex=False)
        
        def normalize_kde(ax, data, i):
            ax2 = ax.twinx()
            # lines = sns.kdeplot(data=data, x="Value", hue="Model", bw_adjust=0.75, clip_on=True, fill=False, linewidth=1.5, legend=False, ax=ax2, palette=self.model_colors, clip=clip).get_lines()
            # if len(lines) == 0:
            #     print("No data")
            #     print('generation', i)  
            #     print(np.var(data["Value"]))
            #     print(prompt)
            #     print(model)
            #     print(measure)
            #     print(data)
                
            x,y = sns.kdeplot(data=data, x="Value",  bw_adjust=0.75, clip_on=True, fill=False, linewidth=1.5, legend=False, ax=ax2,  clip=clip).get_lines()[0].get_data()
            max_density = y.max()
            sns.kdeplot(data=data, x="Value", bw_adjust=0.75, clip_on=True, fill=True, linewidth=1, legend=False, ax=ax2,  alpha=1, clip=clip)
            sns.kdeplot(data=data, x="Value", color = 'black', bw_adjust=0.75, clip_on=True, fill=False, linewidth=linewidth, legend=False, ax=ax2, clip=clip)
            plt.plot([data["Value"].iloc[0], data["Value"].iloc[0]], [0, 1], color='black', linewidth=linewidth)
            
            ax2.set_xlim(clip)

            if i == n_generations - 1:
                ax2.set_xlabel(metric_name)
                ax2.set_xticks(np.linspace(clip[0], clip[1], 3))
                ax.set_xticks(np.linspace(clip[0], clip[1], 3))
            else:
                ax2.set_xticks([])

            xtick_keep = ax2.get_xticks()  
            ax2.set_ylim(0, max_density)
            ax2.set_yticks([])
            ax2.set_ylabel("")
            ax2.set_axis_off()
            ax.set_axis_off()
            return xtick_keep
        
        for i, ax_row in enumerate(g.axes.flat):
            generation_data = ridge_data[ridge_data["Generation"] == i]
            xtick_keep = normalize_kde(ax_row, generation_data, i)
        
        g.fig.subplots_adjust(hspace=-0.8)

        g.axes.flat[-1].set_xlabel(metric_name)
        g.set_titles("")
        g.set(yticks=[])
        g.axes.flat[0].set_ylabel("")
        for i, ax_row in enumerate(g.axes.flat):
            if i == n_generations - 1:
                for tick in xtick_keep:
                    ax_row.text(tick, -0.25, f'{tick:.1f}', ha='center', va='center', fontsize=30, fontweight='bold')
                # mid_pos = (xtick_keep[0] + xtick_keep[-1]) / 2
                # ax_row.text(mid_pos, -0.4, measure.capitalize(), ha='center', va='center', fontsize=35, fontweight='bold')
        if with_suptitle:
            g.fig.suptitle('Convergence', ha='right', fontsize=20, fontweight='bold')

        plt.savefig(f"{saving_name}/metric_distributions_{metric_name}.png", bbox_inches="tight")
        plt.savefig(f"{saving_name}/metric_distributions_{metric_name}.svg", bbox_inches="tight")
        if not no_show:
            plt.show()
        else:
            plt.close()
        print(f"Saved to: {saving_name}/metric_distributions_{metric_name}")
