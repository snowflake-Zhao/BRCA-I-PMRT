import os
import sys
sys.path.append(r'C:\python_apps\demo\demo\\')
import json
import pandas as pd
import torchtuples as tt
from pycox.models import CoxPH
from django.http import HttpResponse

# APP3
def chemo_controller(request):
    jobj = json.loads(request.body)
    in_features = 152
    num_nodes = [20, 32, 25, 21, 23]
    out_features = 1
    batch_norm = True
    dropout = 0.782
    output_bias = False
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)
    model = CoxPH(net, tt.optim.Adam)
    model.load_net(os.getcwd() + "\demo\\chemo_model\surv_model")

    patient = pd.DataFrame(
        {
            "CS tumor size (2004-2015)": 50,
            "Regional nodes positive (1988+)": 1,
            "Regional nodes examined (1988+)": 4,
            "RX Summ--Surg Prim Site (1998+)_0": 0,
            "RX Summ--Surg Prim Site (1998+)_20": 0,
            "RX Summ--Surg Prim Site (1998+)_21": 0,
            "RX Summ--Surg Prim Site (1998+)_22": 0,
            "RX Summ--Surg Prim Site (1998+)_23": 0,
            "RX Summ--Surg Prim Site (1998+)_24": 0,
            "RX Summ--Surg Prim Site (1998+)_30": 0,
            "RX Summ--Surg Prim Site (1998+)_40": 0,
            "RX Summ--Surg Prim Site (1998+)_41": 0,
            "RX Summ--Surg Prim Site (1998+)_42": 0,
            "RX Summ--Surg Prim Site (1998+)_43": 0,
            "RX Summ--Surg Prim Site (1998+)_44": 0,
            "RX Summ--Surg Prim Site (1998+)_45": 0,
            "RX Summ--Surg Prim Site (1998+)_46": 0,
            "RX Summ--Surg Prim Site (1998+)_47": 0,
            "RX Summ--Surg Prim Site (1998+)_48": 0,
            "RX Summ--Surg Prim Site (1998+)_49": 0,
            "RX Summ--Surg Prim Site (1998+)_50": 0,
            "RX Summ--Surg Prim Site (1998+)_51": 1,
            "RX Summ--Surg Prim Site (1998+)_52": 0,
            "RX Summ--Surg Prim Site (1998+)_53": 0,
            "RX Summ--Surg Prim Site (1998+)_54": 0,
            "RX Summ--Surg Prim Site (1998+)_55": 0,
            "RX Summ--Surg Prim Site (1998+)_56": 0,
            "RX Summ--Surg Prim Site (1998+)_57": 0,
            "RX Summ--Surg Prim Site (1998+)_58": 0,
            "RX Summ--Surg Prim Site (1998+)_59": 0,
            "RX Summ--Surg Prim Site (1998+)_60": 0,
            "RX Summ--Surg Prim Site (1998+)_61": 0,
            "RX Summ--Surg Prim Site (1998+)_62": 0,
            "RX Summ--Surg Prim Site (1998+)_63": 0,
            "RX Summ--Surg Prim Site (1998+)_64": 0,
            "RX Summ--Surg Prim Site (1998+)_65": 0,
            "RX Summ--Surg Prim Site (1998+)_66": 0,
            "RX Summ--Surg Prim Site (1998+)_68": 0,
            "RX Summ--Surg Prim Site (1998+)_69": 0,
            "RX Summ--Surg Prim Site (1998+)_71": 0,
            "RX Summ--Surg Prim Site (1998+)_72": 0,
            "RX Summ--Surg Prim Site (1998+)_73": 0,
            "RX Summ--Surg Prim Site (1998+)_74": 0,
            "RX Summ--Surg Prim Site (1998+)_75": 0,
            "RX Summ--Surg Prim Site (1998+)_76": 0,
            "RX Summ--Surg Prim Site (1998+)_80": 0,
            "RX Summ--Surg Prim Site (1998+)_90": 0,
            "RX Summ--Surg Prim Site (1998+)_99": 0,
            "Radiation recode_Beam radiation": 1,
            "Radiation recode_Combination of beam with implants or isotopes": 0,
            "Radiation recode_None/Unknown": 0,
            "Radiation recode_Radiation, NOS  method or source not specified": 0,
            "Radiation recode_Radioactive implants (includes brachytherapy) (1988+)": 0,
            "Radiation recode_Radioisotopes (1988+)": 0,
            "Radiation recode_Recommended, unknown if administered": 0,
            "Radiation recode_Refused (1988+)": 0,
            "Chemotherapy recode (yes, no/unk)_No/Unknown": 0,
            "Chemotherapy recode (yes, no/unk)_Yes": 0,
            "Derived AJCC T, 6th ed (2004-2015)_T1a": 0,
            "Derived AJCC T, 6th ed (2004-2015)_T1b": 0,
            "Derived AJCC T, 6th ed (2004-2015)_T1c": 0,
            "Derived AJCC T, 6th ed (2004-2015)_T1mic": 0,
            "Derived AJCC T, 6th ed (2004-2015)_T2": 1,
            "Derived AJCC N, 6th ed (2004-2015)_N1": 0,
            "Derived AJCC N, 6th ed (2004-2015)_N1NOS": 0,
            "Derived AJCC N, 6th ed (2004-2015)_N1a": 0,
            "Derived AJCC N, 6th ed (2004-2015)_N1b": 0,
            "Derived AJCC N, 6th ed (2004-2015)_N1c": 0,
            "Derived AJCC N, 6th ed (2004-2015)_N1mi": 1,
            "Derived AJCC M, 6th ed (2004-2015)_M0": 1,
            "ICD-O-3 Hist/behav_8000/3: Neoplasm, malignant": 0,
            "ICD-O-3 Hist/behav_8010/3: Carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8013/3: Large cell neuroendocrine carcinoma": 0,
            "ICD-O-3 Hist/behav_8020/3: Carcinoma, undifferentiated, NOS": 0,
            "ICD-O-3 Hist/behav_8021/3: Carcinoma, anaplastic, NOS": 0,
            "ICD-O-3 Hist/behav_8022/3: Pleomorphic carcinoma": 0,
            "ICD-O-3 Hist/behav_8030/3: Giant cell and spindle cell carcinoma": 0,
            "ICD-O-3 Hist/behav_8035/3: Carcinoma with osteoclast-like giant cells": 0,
            "ICD-O-3 Hist/behav_8041/3: Small cell carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8050/3: Papillary carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8070/3: Squamous cell carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8140/3: Adenocarcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8200/3: Adenoid cystic carcinoma": 0,
            "ICD-O-3 Hist/behav_8201/3: Cribriform carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8211/3: Tubular adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8230/3: Solid carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8246/3: Neuroendocrine carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8251/3: Alveolar adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8255/3: Adenocarcinoma with mixed subtypes": 0,
            "ICD-O-3 Hist/behav_8260/3: Papillary adenocarcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8290/3: Oxyphilic adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8343/3: Papillary carcinoma, encapsulated": 0,
            "ICD-O-3 Hist/behav_8401/3: Apocrine adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8453/3: Intraductal papillary-mucinous carcinoma, invasive": 0,
            "ICD-O-3 Hist/behav_8480/3: Mucinous adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8481/3: Mucin-producing adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8490/3: Signet ring cell carcinoma": 0,
            "ICD-O-3 Hist/behav_8500/3: Infiltrating duct carcinoma, NOS": 1,
            "ICD-O-3 Hist/behav_8501/3: Comedocarcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8502/3: Secretory carcinoma of breast": 0,
            "ICD-O-3 Hist/behav_8503/3: Intraductal papillary adenocarcinoma with invasion": 0,
            "ICD-O-3 Hist/behav_8504/3: Intracystic carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8507/3: Ductal carcinoma, micropapillary": 0,
            "ICD-O-3 Hist/behav_8510/3: Medullary carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8513/3: Atypical medullary carcinoma": 0,
            "ICD-O-3 Hist/behav_8520/3: Lobular carcinoma, NOS": 0,
            "ICD-O-3 Hist/behav_8521/3: Infiltrating ductular carcinoma": 0,
            "ICD-O-3 Hist/behav_8522/3: Infiltrating duct and lobular carcinoma": 0,
            "ICD-O-3 Hist/behav_8523/3: Infiltrating duct mixed with other types of carcinoma": 0,
            "ICD-O-3 Hist/behav_8524/3: Infiltrating lobular mixed with other types of carcinoma": 0,
            "ICD-O-3 Hist/behav_8525/3: Polymorphous low grade adenocarcinoma": 0,
            "ICD-O-3 Hist/behav_8540/3: Paget disease, mammary": 0,
            "ICD-O-3 Hist/behav_8541/3: Paget disease and infiltrating ductal carcinoma of breast": 0,
            "ICD-O-3 Hist/behav_8543/3: Paget disease and intraductal carcinoma": 0,
            "ICD-O-3 Hist/behav_8550/3: Acinar cell carcinoma": 0,
            "ICD-O-3 Hist/behav_8560/3: Adenosquamous carcinoma": 0,
            "ICD-O-3 Hist/behav_8574/3: Adenocarcinoma with neuroendocrine differentiation": 0,
            "ICD-O-3 Hist/behav_8575/3: Metaplastic carcinoma, NOS": 0,
            "Age recode with <1 year olds_15-19 years": 0,
            "Age recode with <1 year olds_20-24 years": 0,
            "Age recode with <1 year olds_25-29 years": 0,
            "Age recode with <1 year olds_30-34 years": 0,
            "Age recode with <1 year olds_35-39 years": 0,
            "Age recode with <1 year olds_40-44 years": 0,
            "Age recode with <1 year olds_45-49 years": 0,
            "Age recode with <1 year olds_50-54 years": 0,
            "Age recode with <1 year olds_55-59 years": 0,
            "Age recode with <1 year olds_60-64 years": 0,
            "Age recode with <1 year olds_65-69 years": 0,
            "Age recode with <1 year olds_70-74 years": 0,
            "Age recode with <1 year olds_75-79 years": 0,
            "Age recode with <1 year olds_80-84 years": 0,
            "Age recode with <1 year olds_85+ years": 0,
            "Breast Subtype (2010+)_HR+/HER2+ (Luminal B)": 0,
            "Breast Subtype (2010+)_HR+/HER2- (Luminal A)": 1,
            "ER Status Recode Breast Cancer (1990+)_Negative": 0,
            "ER Status Recode Breast Cancer (1990+)_Positive": 1,
            "PR Status Recode Breast Cancer (1990+)_Negative": 0,
            "PR Status Recode Breast Cancer (1990+)_Positive": 1,
            "Marital status at diagnosis_Divorced": 0,
            "Marital status at diagnosis_Married (including common law)": 1,
            "Marital status at diagnosis_Separated": 0,
            "Marital status at diagnosis_Single (never married)": 0,
            "Marital status at diagnosis_Unmarried or Domestic Partner": 0,
            "Marital status at diagnosis_Widowed": 0,
            "Grade (thru 2017)_Moderately differentiated; Grade II": 0,
            "Grade (thru 2017)_Poorly differentiated; Grade III": 0,
            "Grade (thru 2017)_Undifferentiated; anaplastic; Grade IV": 1,
            "Grade (thru 2017)_Well differentiated; Grade I": 0,
            "Laterality_Bilateral, single primary": 0,
            "Laterality_Left - origin of primary": 0,
            "Laterality_Right - origin of primary": 1
        }, index=[0]
    )

    # CS tumor size
    tumorSize = int(jobj["tumorSize"])
    patient.iloc[0, patient.columns.get_loc("CS tumor size (2004-2015)")] = tumorSize

    # Regional nodes positive
    PRN = int(jobj["PRN"])
    patient.iloc[0, patient.columns.get_loc("Regional nodes positive (1988+)")] = PRN

    # Regional nodes examined
    ERN = int(jobj["ERN"])
    patient.iloc[0, patient.columns.get_loc("Regional nodes examined (1988+)")] = ERN

    # RX Summ--Surg Prim Site (1998+)
    primSurg = int(jobj["primSurg"])
    if primSurg == 1:
        patient.iloc[0, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_22")] = 1
    elif primSurg == 2:
        patient.iloc[0, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_23")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_51")] = 1

    # Chemotherapy recode (yes, no/unk)
    chemotherapy = int(jobj["chemotherapy"])
    if chemotherapy == 1:
        patient.iloc[0, patient.columns.get_loc("Chemotherapy recode (yes, no/unk)_No/Unknown")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Chemotherapy recode (yes, no/unk)_Yes")] = 1

    # Derived AJCC T, 6th ed (2004-2015)
    T = int(jobj["T"])
    if T == 1:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC T, 6th ed (2004-2015)_T2")] = 1
    elif T == 2:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC T, 6th ed (2004-2015)_T1c")] = 1
    elif T == 3:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC T, 6th ed (2004-2015)_T1b")] = 1
    elif T == 4:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC T, 6th ed (2004-2015)_T1a")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC T, 6th ed (2004-2015)_T1mic")] = 1

    # Derived AJCC N, 6th ed (2004-2015)
    N = int(jobj["N"])
    if N == 1:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC N, 6th ed (2004-2015)_N1a")] = 1
    elif N == 2:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC N, 6th ed (2004-2015)_N1mi")] = 1
    elif N == 3:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC N, 6th ed (2004-2015)_N1")] = 1
    elif N == 4:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC N, 6th ed (2004-2015)_N1NOS")] = 1
    elif N == 5:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC N, 6th ed (2004-2015)_N1c")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Derived AJCC N, 6th ed (2004-2015)_N1b")] = 1

    # Derived AJCC M, 6th ed (2004-2015)
    patient.iloc[0, patient.columns.get_loc("Derived AJCC M, 6th ed (2004-2015)_M0")] = 1

    # ICD-O-3 Hist/behav
    Hist = int(jobj["Hist"])
    if Hist == 1:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8520/3: Lobular carcinoma, NOS")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8500/3: Infiltrating duct carcinoma, NOS")] = 1

    # age
    age = int(jobj["ageRecode"])
    if age == 1:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_15-19 years")] = 1
    elif age == 2:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_15-19 years")] = 1
    elif age == 3:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_25-29 years")] = 1
    elif age == 4:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_30-34 years")] = 1
    elif age == 5:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_35-39 years")] = 1
    elif age == 6:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_40-44 years")] = 1
    elif age == 7:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_45-49 years")] = 1
    elif age == 8:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_50-54 years")] = 1
    elif age == 9:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_55-59 years")] = 1
    elif age == 10:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_60-64 years")] = 1
    elif age == 11:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_65-69 years")] = 1
    elif age == 12:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_70-74 years")] = 1
    elif age == 13:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_75-79 years")] = 1
    elif age == 14:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_80-84 years")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_85+ years")] = 1

    # Breast Subtype (2010+)
    breastSubtype = int(jobj["breastSubtype"])
    if breastSubtype == 1:
        patient.iloc[0, patient.columns.get_loc("Breast Subtype (2010+)_HR+/HER2- (Luminal A)")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Breast Subtype (2010+)_HR+/HER2+ (Luminal B)")] = 1

    # ER Status Recode Breast Cancer (1990+)
    ERStatus = int(jobj["ERStatus"])
    if ERStatus == 1:
        patient.iloc[0, patient.columns.get_loc("ER Status Recode Breast Cancer (1990+)_Positive")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("ER Status Recode Breast Cancer (1990+)_Negative")] = 1

    # PR Status Recode Breast Cancer (1990+)
    PRStatus = int(jobj["PRStatus"])
    if PRStatus == 1:
        patient.iloc[0, patient.columns.get_loc("PR Status Recode Breast Cancer (1990+)_Positive")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("PR Status Recode Breast Cancer (1990+)_Negative")] = 1

    # Marital status at diagnosis
    maritalStatus = int(jobj["maritalStatus"])
    if maritalStatus == 1:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Single (never married)")] = 1
    elif maritalStatus == 2:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Married (including common law)")] = 1
    elif maritalStatus == 3:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Divorced")] = 1
    elif maritalStatus == 4:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Widowed")] = 1
    elif maritalStatus == 5:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Separated")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Unmarried or Domestic Partner")] = 1

    # Grade (thru 2017)
    grade = int(jobj["grade"])
    if grade == 1:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Well differentiated; Grade I")] = 1
    elif grade == 2:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Moderately differentiated; Grade II")] = 1
    elif grade == 3:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Poorly differentiated; Grade III")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Undifferentiated; anaplastic; Grade IV")] = 1

    # Laterality
    laterality = int(jobj["laterality"])
    if laterality == 1:
        patient.iloc[0, patient.columns.get_loc("Laterality_Left - origin of primary")] = 1
    else:
        patient.iloc[0, patient.columns.get_loc("Laterality_Right - origin of primary")] = 1

    patient = pd.concat([patient] * 2, ignore_index=False)
    patient.iloc[0, patient.columns.get_loc("Radiation recode_None/Unknown")] = 0
    patient.iloc[0, patient.columns.get_loc("Radiation recode_Beam radiation")] = 1

    patient.iloc[1, patient.columns.get_loc("Radiation recode_None/Unknown")] = 1
    patient.iloc[1, patient.columns.get_loc("Radiation recode_Beam radiation")] = 0

    pre_haz_rate = model.predict_surv_df(patient.to_numpy().astype('float32'))
    line_series = []
    for i in range(2):
        line_series.append(pre_haz_rate.iloc[:, i].tolist())
    res = {
        "series": line_series
    }
    return HttpResponse(json.dumps(res), content_type="application/json")
