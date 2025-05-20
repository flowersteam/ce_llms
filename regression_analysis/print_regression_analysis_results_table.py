import json

def p_value_to_stars_string(p_value):
    stars = ''
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    return stars

# Load data
deps = ["q", "d"]
datasets = ["all", "webis_reddit", "100m_tweets", "reddit_submissions", "wikipedia"]
gen_n = ["500", "1000"]

data = {}
all_coefs = set()

for dep in deps:
    data[dep] = {}
    for dataset in datasets:
        data[dep][dataset] = {}
        if dataset == 'all':
            path = f"regression_analysis/results/all/ols_{dep}_[500, 1000].json"
            with open(path, 'r') as f:
                regression_results = json.load(f)

            data[dep][dataset]["all"] = regression_results.copy()

        else:
            for g in gen_n:
                data[dep][dataset][g] = {}
                path = f"regression_analysis/results/{dataset}_clusters/ols_{dep}_{g}.json"

                with open(path, 'r') as f:
                    regression_results = json.load(f)

                data[dep][dataset][g] = regression_results.copy()
                all_coefs.update(data[dep][dataset][g]['coefs'].keys())


import pandas as pd

for dep in deps:
    print(f"{'-' * 40}")
    print(f"\n{'Quality' if dep == 'q' else 'Diversity'}")
    print(f"{'-' * 40}")

    # Build DataFrame
    rows = []
    for coef in sorted(all_coefs):
        if coef in ["const", "intercept"]:
            continue

        row = {"coefficient": coef.split("_cap_")[0]}
        for dataset in datasets:
            if dataset == "all":
                key = f"{dataset}"
                c = data[dep][dataset]['all']['coefs'][coef]
                p = data[dep][dataset]['all']['p_values'][coef]
                row[key] = f"{c:.4f}{p_value_to_stars_string(p)}"
            else:
                for g in gen_n:
                    key = f"{dataset} (r:1/{int(4000/int(g))})"
                    c = data[dep][dataset][g]['coefs'][coef]
                    p = data[dep][dataset][g]['p_values'][coef]
                    row[key] = f"{p_value_to_stars_string(p)}{c:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows).set_index("coefficient")
    print(df.to_string(float_format="{:.5}".format))
