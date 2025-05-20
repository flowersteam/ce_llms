python -m regression_analysis.regression_analysis_merge d --partition webis_reddit  --no-show --save-dir regression_analysis/results/merged
pdfcrop --margins '0 0 -500 0' regression_analysis/results/merged/ols_d_webis_reddit_\[500\,\ 1000\]_labs.pdf regression_analysis/results/merged/ols_labels.pdf
for path in regression_analysis/results/merged/*.pdf; do convert "$path" "${path%.pdf}.png"; done
for path in regression_analysis/results/merged/*.pdf; do convert "$path" "${path%.pdf}.svg"; done
exit
python -m regression_analysis.regression_analysis_merge d --partition webis_reddit --no-show --save-dir regression_analysis/results/merged --no-labels
python -m regression_analysis.regression_analysis_merge d --partition 100m_tweets  --no-show --save-dir regression_analysis/results/merged --no-labels
python -m regression_analysis.regression_analysis_merge d --partition wikipedia    --no-show --save-dir regression_analysis/results/merged --no-labels
#
python -m regression_analysis.regression_analysis_merge q --partition webis_reddit --no-show --save-dir regression_analysis/results/merged --no-labels
python -m regression_analysis.regression_analysis_merge q --partition 100m_tweets  --no-show --save-dir regression_analysis/results/merged --no-labels
python -m regression_analysis.regression_analysis_merge q --partition wikipedia    --no-show --save-dir regression_analysis/results/merged --no-labels

for path in regression_analysis/results/merged/*.pdf; do convert "$path" "${path%.pdf}.png"; done
for path in regression_analysis/results/merged/*.pdf; do convert "$path" "${path%.pdf}.svg"; done

#################
exit


## webis
#for dep in d q; do
#    for gen in 500 1000; do
#        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
#          --save-dir regression_analysis/results/webis_reddit_clusters \
#          --clusters-indices-to-path-json data/webis/selected_clusters_indices_to_path.json \
#          --clusters-evaluation-dir data/webis/webis_dataset_clusters/evaluation \
#          --clusters-simulation-results-dir eval_results/simulation_results/webis_reddit_clusters
#
#    done
#
#done
#for path in regression_analysis/results/webis_reddit_clusters/*.pdf; do convert $path ${path%.pdf}.png; done
#
## 100m tweets
#for dep in d q; do
#    for gen in 500 1000; do
#         echo ""
#        # OLS
#        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
#            --save-dir regression_analysis/results/100m_tweets_clusters \
#            --clusters-indices-to-path-json data/twitter_100m/selected_clusters_indices_to_path.json \
#            --clusters-evaluation-dir data/twitter_100m/100m_tweets_dataset_clusters/evaluation \
#            --clusters-simulation-results-dir eval_results/simulation_results/100m_tweets_clusters
#
#
#    done
#done
#for path in regression_analysis/results/100m_tweets_clusters/*.pdf; do convert $path ${path%.pdf}.png; done
#
### reddit submissions
#for dep in d q; do
#    for gen in 500 1000; do
#      echo ""
#        # OLS
#        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
#            --save-dir regression_analysis/results/reddit_submissions_clusters \
#            --clusters-indices-to-path-json data/reddit_submissions/selected_clusters_indices_to_path.json \
#            --clusters-evaluation-dir data/reddit_submissions/reddit_submissions_dataset_clusters/evaluation \
#            --clusters-simulation-results-dir eval_results/simulation_results/reddit_submissions_clusters
#
#    done
#
#done
#for path in regression_analysis/results/reddit_submissions_clusters/*.pdf; do convert $path ${path%.pdf}.png; done
#
## wikipedia
#for dep in d q; do
#    for gen in 500 1000; do
#        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
#          --save-dir regression_analysis/results/wikipedia_clusters \
#          --clusters-indices-to-path-json data/wikipedia/selected_clusters_indices_to_path.json \
#          --clusters-evaluation-dir data/wikipedia/wikipedia_dataset_clusters/evaluation \
#          --clusters-simulation-results-dir eval_results/simulation_results/wikipedia_clusters
#
#    done
#
#done
#for path in regression_analysis/results/wikipedia_clusters/*.pdf; do convert $path ${path%.pdf}.png; done


#python -m regression_analysis.regression_analysis_all d --save-dir regression_analysis/results/all --no-show -gs 1000
#python -m regression_analysis.regression_analysis_all q --save-dir regression_analysis/results/all --no-show -gs 1000
#
#python -m regression_analysis.regression_analysis_all d --save-dir regression_analysis/results/all --no-show -gs 500
#python -m regression_analysis.regression_analysis_all q --save-dir regression_analysis/results/all --no-show -gs 500

python -m regression_analysis.regression_analysis_all d --save-dir regression_analysis/results/all --no-show -gs 500 1000
python -m regression_analysis.regression_analysis_all q --save-dir regression_analysis/results/all --no-show -gs 500 1000

for path in regression_analysis/results/all/*.pdf; do convert "$path" "${path%.pdf}.png"; done
