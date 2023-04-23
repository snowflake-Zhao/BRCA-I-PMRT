import pandas as pd
import pycox
import torchtuples

import utilities
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler



df_train = pd.read_csv("dataset\\training_data.csv")
df_test = pd.read_csv("dataset\\test_data.csv", cache_dates=False)

model_path = "model\\surv1"
df_test_count = utilities.get_data_frame_row_count(df_test)

df = df_test.append(df_train)

df = utilities.remove_col(df, "Sex")

df = pd.get_dummies(df, prefix=["RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)",
                                "Derived AJCC T, 6th ed (2004-2015)",
                                "Derived AJCC N, 6th ed (2004-2015)",
                                "Derived AJCC M, 6th ed (2004-2015)", "ICD-O-3 Hist/behav",
                                "Age recode with <1 year olds", "Breast Subtype (2010+)",
                                "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
                                "Marital status at diagnosis", "Grade (thru 2017)",
                                "Laterality"],
                    columns=["RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                             "Chemotherapy recode (yes, no/unk)",
                             "Derived AJCC T, 6th ed (2004-2015)",
                             "Derived AJCC N, 6th ed (2004-2015)",
                             "Derived AJCC M, 6th ed (2004-2015)", "ICD-O-3 Hist/behav",
                             "Age recode with <1 year olds", "Breast Subtype (2010+)",
                             "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
                             "Marital status at diagnosis", "Grade (thru 2017)",
                             "Laterality"])
df["End Calc Vital Status (Adjusted)"] = df["End Calc Vital Status (Adjusted)"].apply(utilities.encode_event)

df_train = df[df_test_count:]
df_test = df[:df_test_count]

df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

y_train = utilities.get_surv_target(df_train)
y_val = utilities.get_surv_target(df_val)

durations_test, events_test = utilities.get_surv_target(df_test)

x_train = utilities.remove_col(utilities.remove_col(df_train, "Number of Intervals (Calculated)"),
                               "End Calc Vital Status (Adjusted)")
x_val = utilities.remove_col(utilities.remove_col(df_val, "Number of Intervals (Calculated)"),
                             "End Calc Vital Status (Adjusted)")
x_test = utilities.remove_col(utilities.remove_col(df_test, "Number of Intervals (Calculated)"),
                              "End Calc Vital Status (Adjusted)")

cols_standardize = []
cols_leave = list(x_train.keys())
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(x_train).astype('float32')
x_val = x_mapper.transform(x_val).astype('float32')
x_test = x_mapper.transform(x_test).astype('float32')
val = x_val, y_val

net = torchtuples.practical.MLPVanilla(x_train.shape[1], [20, 32, 25, 21, 23], 1, True, 0.782,
                                       output_bias=False)
model = pycox.models.CoxPH(net, torchtuples.optim.Adam)
model.load_net("model\surv_model")
surv = model.predict_surv_df(x_test)
eval_surv = pycox.evaluation.EvalSurv(surv, durations_test, events_test, censor_surv='km')

# The concordance index of this model is : 0.769179004037685
print(utilities.format_string(
    "The concordance index of this model is : {0}",
    [eval_surv.concordance_td()]))