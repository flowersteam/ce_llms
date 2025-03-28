
# load the results from the clustering results
import pickle
import numpy as np

# go over files in clustering_results
import glob
pickle_files = glob.glob("./clustering_results/*.pkl")
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        metrics = pickle.load(f)
        print("Cluster results loaded from:", pickle_file)

    aggreagated_metrics = {}
    for m in metrics:
        try:
            aggreagated_metrics[m] = np.mean(metrics[m])
        except:
            if m == "cluster_path":
                aggreagated_metrics[m] = metrics[m]
            elif m == "text_cap_80000":
                ...
            elif m == "text_cap_250":
                aggreagated_metrics[m] = metrics[m]
            else:
                raise ValueError("Unknown metric:", m)

    pickle_file = pickle_file.replace("clustering_results", "aggregated_clustering_results")

    # with open(pickle_file, 'wb') as f:
    #     pickle.dump(aggreagated_metrics, f)
    #     print("Cluster results saved to:", pickle_file)
