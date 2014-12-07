Predict whether a certain advertisement will be clicked, based on certain features like

- hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
- C1 -- anonymized categorical variable
- banner_pos
- site_id
- site_domain
- site_category
- app_id
- app_domain
- app_category
- device_id
- device_ip
- device_model
- device_type
- device_conn_type
- C14-C21 -- anonymized categorical variables

Link: https://www.kaggle.com/c/avazu-ctr-prediction/
Download Data: https://www.kaggle.com/c/avazu-ctr-prediction/
(Only download train.gz)

We use training examples to generate test examples.
For example, the first 1000000 lines are used as training examples and the next
remaining lines are used as test examples.

Processed files:
- https://www.dropbox.com/s/84jqkc29vfchak8/train_1000000_lines?dl=0
- https://www.dropbox.com/s/1z4el78naem08rt/test_100000_lines?dl=0
- https://www.dropbox.com/s/vutubfocxrg1mz7/sample_100?dl=0

Use train_10000000 to train the classifier, and use the classifier on
test_100000 to see how accurate the classifier predicts the label.
Right now the accuracy is about 80%.