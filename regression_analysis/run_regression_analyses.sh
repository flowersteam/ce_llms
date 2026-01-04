echo "MERGED"
echo div webis
python -m regression_analysis.regression_analysis_merge d --partition webis_reddit --no-show --save-dir regression_analysis/results/merged --no-labels   | grep "R-squared"
echo div 100m_tweets
python -m regression_analysis.regression_analysis_merge d --partition 100m_tweets  --no-show --save-dir regression_analysis/results/merged --no-labels | grep "R-squared"
echo div wikipedia
python -m regression_analysis.regression_analysis_merge d --partition wikipedia    --no-show --save-dir regression_analysis/results/merged --no-labels | grep "R-squared"
#
echo quality webis
python -m regression_analysis.regression_analysis_merge q --partition webis_reddit --no-show --save-dir regression_analysis/results/merged --no-labels | grep "R-squared"
echo quality 100m_tweets
python -m regression_analysis.regression_analysis_merge q --partition 100m_tweets  --no-show --save-dir regression_analysis/results/merged --no-labels | grep "R-squared"
echo quality wikipedia
python -m regression_analysis.regression_analysis_merge q --partition wikipedia    --no-show --save-dir regression_analysis/results/merged --no-labels | grep "R-squared"

#for path in regression_analysis/results/merged/*.pdf; do convert "$path" "${path%.pdf}.png"; done
#for path in regression_analysis/results/merged/*.pdf; do convert "$path" "${path%.pdf}.svg"; done

#################

echo "STANDARD"
# webis
for dep in d q; do
    for gen in 500 1000; do
        echo webis $dep $gen
        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
          --save-dir regression_analysis/results/webis_reddit_clusters \
          --clusters-indices-to-path-json data/webis/selected_clusters_indices_to_path.json \
          --clusters-evaluation-dir data/webis/webis_dataset_clusters/evaluation \
          --clusters-simulation-results-dir eval_results/simulation_results/webis_reddit_clusters  | grep "R-squared"

    done

done
#for path in regression_analysis/results/webis_reddit_clusters/*.pdf; do convert $path ${path%.pdf}.png; done

# 100m tweets
for dep in d q; do
    for gen in 500 1000; do
        echo 100m $dep $gen
        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
            --save-dir regression_analysis/results/100m_tweets_clusters \
            --clusters-indices-to-path-json data/twitter_100m/selected_clusters_indices_to_path.json \
            --clusters-evaluation-dir data/twitter_100m/100m_tweets_dataset_clusters/evaluation \
            --clusters-simulation-results-dir eval_results/simulation_results/100m_tweets_clusters | grep "R-squared"


    done
done
#for path in regression_analysis/results/100m_tweets_clusters/*.pdf; do convert $path ${path%.pdf}.png; done

## reddit submissions
for dep in d q; do
    for gen in 500 1000; do
        echo reddit_sub $dep $gen
        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
            --save-dir regression_analysis/results/reddit_submissions_clusters \
            --clusters-indices-to-path-json data/reddit_submissions/selected_clusters_indices_to_path.json \
            --clusters-evaluation-dir data/reddit_submissions/reddit_submissions_dataset_clusters/evaluation \
            --clusters-simulation-results-dir eval_results/simulation_results/reddit_submissions_clusters | grep "R-squared"

    done

done
#for path in regression_analysis/results/reddit_submissions_clusters/*.pdf; do convert $path ${path%.pdf}.png; done

# wikipedia
for dep in d q; do
    for gen in 500 1000; do
        echo wikipedia $dep $gen
        python -m regression_analysis.regression_analysis $dep -g $gen --no-show --cv 10 \
          --save-dir regression_analysis/results/wikipedia_clusters \
          --clusters-indices-to-path-json data/wikipedia/selected_clusters_indices_to_path.json \
          --clusters-evaluation-dir data/wikipedia/wikipedia_dataset_clusters/evaluation \
          --clusters-simulation-results-dir eval_results/simulation_results/wikipedia_clusters | grep "R-squared"

    done

done
#for path in regression_analysis/results/wikipedia_clusters/*.pdf; do convert $path ${path%.pdf}.png; done

echo "ALL"
echo all div
python -m regression_analysis.regression_analysis_all d --save-dir regression_analysis/results/all --no-show -gs 500 1000 | grep "R-squared"
echo all q
python -m regression_analysis.regression_analysis_all q --save-dir regression_analysis/results/all --no-show -gs 500 1000 | grep "R-squared"

#for path in regression_analysis/results/all/*.pdf; do convert "$path" "${path%.pdf}.png"; done
