# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was created by Danny as part of the WGU D501 Machine Learning DevOps project. It is a scikit-learn (version 1.5.1) `RandomForestClassifier` with 100 trees, a maximum tree depth of 15, and `random_state=42` so that results can be reproduced. The eight categorical features are one-hot encoded with a `OneHotEncoder`, and the label is binarized with a `LabelBinarizer` using the `process_data` function in `ml/data.py`. The trained model and encoder are saved as pickle files in the `model/` folder and are served through a FastAPI application in `main.py`.

## Intended Use

The model predicts whether a person's annual income is more than $50K or at most $50K based on demographic and employment attributes from census data. It is intended for educational use, to demonstrate how a model can be trained, tested on data slices, documented, and deployed through a REST API with continuous integration. It is not intended to be used for real decisions about individuals, such as hiring, lending, insurance, or eligibility for services.

## Training Data

The data is the Census Income ("Adult") dataset from the UCI Machine Learning Repository, which was extracted from the 1994 U.S. Census database. The provided `census.csv` file contains 32,561 rows and 15 columns: six numeric features (age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week), eight categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country), and the `salary` label. About 76% of the rows are labeled `<=50K` and about 24% are labeled `>50K`. Missing values in the original data appear as the string `?` and were kept as their own category. The data was split with `train_test_split` into 80% training data (26,048 rows), stratified on the label with `random_state=42`.

## Evaluation Data

The evaluation data is the remaining 20% of `census.csv` (6,513 rows) from the same stratified split. It was processed with the encoder and label binarizer that were fitted on the training data, so no information from the test set was used during training.

## Metrics

The model was evaluated with precision, recall, and F1 score, where the positive class is `>50K`. On the test set, the model achieved a precision of 0.7932, a recall of 0.5918, and an F1 score of 0.6779. This means that when the model predicts `>50K` it is correct about 79% of the time, but it finds only about 59% of the people who actually earn more than $50K.

Performance was also measured on every value of every categorical feature, and the full results are saved in `slice_output.txt`. For example, the F1 score was 0.6617 for the `sex: Female` slice and 0.6806 for the `sex: Male` slice, 0.7932 for `education: Bachelors` and 0.3853 for `education: HS-grad`, and 0.6818 for `race: White` and 0.6275 for `race: Black`.

## Ethical Considerations

The dataset contains sensitive attributes such as race, sex, marital status, and native country, and the model uses them as features. The slice results show that performance is not equal across groups, so the model could treat some groups less fairly than others if it were used for real decisions. Several slices have very few rows (for example, some native-country values have fewer than five people), so their metrics are unreliable and should not be read as evidence that the model works well or poorly for those groups. The data is also from 1994, so it reflects the income levels and social patterns of that time.

## Caveats and Recommendations

The $50K threshold is not adjusted for inflation, so the predictions do not describe incomes today. The model's recall is noticeably lower than its precision, which means it misses many higher-income people, especially in slices such as HS-grad and Some-college. Future work could tune hyperparameters with cross-validation, try class weighting to improve recall, test removing the sensitive features, and retrain on more recent census data. Anyone who reuses the model should review the slice metrics before relying on its predictions for any group.
