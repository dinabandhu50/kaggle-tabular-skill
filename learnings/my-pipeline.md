Datascience Framework I operate on:
---
0. The project structure setup and env variable setups 
    - setting up folder structure like data, configs, src, notebooks, scripts, submissions, wandb, experiments (to keep track of exp and feature engineering and notes), localdev for small experiment and own testing, 
    - do not forget to update .gitignore to ignore appropriately
    - use `uv` for environment setup and management
    - user `.envrc` and `justfile` for repeated commands and env vars.
    - settsetting up env vars such as kaggle and wandb api keys.
1. Create the baseline end to end process from downloading kaggle dataset to local dir, to baseline submission to kaggle.
2. ROI based efforts
    - Remember the priority and ROI
    - The feature engineering can improve score in 10s range like very high ROI, 
    - The hyperparameter tuning is in 5s range i.e. 2nd highest ROI 
    - The model ensemble gives us the 1-2% improvements the performance squeezer last hope of ROI.
    - We need to put our effort accordingly

3. Experiment tracking and improvements
    - even though we have wandb for our exp tracker, we should make notes of what all changes we did in our `experiments` folder based on models like `experiments/xgb` or something like this.
    - write markdown file and include what feature engineering was done and what worked and what did not worked, the analysis and findings of that experiemnts.
    - 
3. Experiemnt order and details
    - First we do feature engineering, we should use some well known feature engineering methods used by ML experts to improve the base model performance scores.
    - Do an hypothesis backed by mathmetical thinking and intuition and generate features based on deep thinking on each features, we should not blindly generate many features we should have some thinking behind each engineered features. 
    - After generating few engineered features we should test locally the performance of model, keep fetaures which are improving the model performance scores in the fe group and make note and do analysis. 
    - This feature engineering is an iterative process so no need to try to do it at once, do systematically and plan accordingly.
    - After obtaining batch of successful engineeried features and teh performance is plauteing, we can go for hyperparameter tuning.
    - In model hyperparameter tuning we should do hand tuning 1st based on intuition and logic, and check if that is working, and do the optuna or hyperopt based automatic hyperparameter based tuning on a small grid for faster and better results, this is how I do the experiemnts, but this is up to you if you wanna start with automatic hyperparameter tuning you can start also.
    - after obtaining the final best model we go for submission and check the performance and do analysis like what is happening for exampple, e.g. is the model overfitting or underfitting, and based on the findings we plan experiments and feature engineering and hyperparameter tuning again.
    - we must do this for multiple models such as xgb, lgbm, catboost, rf etc. we should do separate and save the details related to each models, like for feature engineering for each model and what all features good for this keep the oofs predictions for future model ensembling, later it would be very helpful, also for debugging and analysis purpose also, the systematic tracking and keeping all details and ideas for each models will help us in improvements. 
    - Finally when we have some best single models and the performance is not improving much even after feature engineering or hyperparameter tuning, we should go for model ensembliong, only after getting all three best models we go for model ensembles. 
    - In model ensembling we use oofs and trained model for further improvements. This is prone to overfitting so please be very cautious. Finally submit the performance and check where we are. and try to gather more information from kaggle discussion and from web and research papers about the features which works, the methods which can work and try to implement them in our experiments.

please make a plan accordingly and start one by one model, or you can also spin multiple subagent to do the best model fining parallely. that is up to you which one is good and easy for you do that.
