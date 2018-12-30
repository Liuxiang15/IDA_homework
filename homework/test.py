import pandas as pd
test_data = pd.Series(['Caucasian', 'Male', '[80-90)', 1, 6, 7, 6, 60, 2, 21, 1, 0, 0],
                          index=['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
                                 'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                                 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient'])
print(test_data[0])