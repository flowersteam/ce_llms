#python regression_analysis_supression.py q -g 500 --cv 10 --no-show
#python regression_analysis_supression.py q -g 1000 --cv 10 --no-show
#python regression_analysis_supression.py d -g 500 --cv 10 --no-show
#python regression_analysis_supression.py d -g 1000 --cv 10 --no-show

#python regression_analysis.py q -g 500 --no-show --cv 10
#python regression_analysis.py q -g 1000 --no-show --cv 10
#python regression_analysis.py d -g 500 --no-show --cv 10
#python regression_analysis.py d -g 1000 --no-show --cv 10 --legend
#
python regression_analysis_plsr_old.py d -g 500 --no-show --cv 10 --no-feature-selection
python regression_analysis_plsr_old.py d -g 1000 --no-show --cv 10 --no-feature-selection
python regression_analysis_plsr_old.py q -g 500 --no-show --cv 10 --no-feature-selection
python regression_analysis_plsr_old.py q -g 1000 --no-show --cv 10 --no-feature-selection