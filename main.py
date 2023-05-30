import joblib
import numpy as np
import joblib
import json
import os 
import warnings
warnings.filterwarnings("ignore")   
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

curr_dir = os.path.dirname(os.path.abspath(__file__))
encoded_data = pd.read_csv(curr_dir+'/X_train.csv')

model = joblib.load(curr_dir+'/models/GradientBoostingClassifier.joblib')
scaler = joblib.load(curr_dir+'/models/mnmx_scaler.joblib')
label_encoder = joblib.load(curr_dir+'/models/label_encoder.joblib')

code_gender =['Z794', 'I5033', 'E10649', 'R4182', 'R42', 'R918', 'H2513', 'I10',
       'E7800', 'Z888', 'R310', 'Z825', 'Z90710', 'J441', 'D122',
       'Z87891', 'I252', 'D125', 'Z8546', 'I480', 'G8929', 'M19022',
       'F419', 'Z01812', 'R079', 'M5126', 'N179']
code_age = ['R079', 'E873', 'I10', 'I25810', 'I440', 'K5900', 'S4992XA',
       'M5137', 'G459', 'R0789', 'E785', 'R928', 'S42212A', 'Z90710',
       'I482', 'I739', 'I70229', 'E8342', 'F329', 'M79642', 'E806',
       'I071', 'Z23', 'M19072', 'E875', 'I4891', 'E7800']

#Frequency based imputation
hcpcs_codes = ["87040",	"82306",	"84484",	"71048",	"97164",	"83735",	"85018",	"Q0510",	"99100",	"85025",	"P2028",	"Q9967",	"85027",	"G0378",	"93229",	"80048",	"80053",	"80061",	"36415",	"A4206",	"Z7512",	"93005",	"A9270",	"99284",	"97110",	"84439",	"84443",	"83036",	"J7030",	"93798",	"96360",	"81001",	"85610",	"81003",	"88299",	"G0463",	"97140",	"82550",	"97530",	"J3010"]
counties =["Butler","Bailey","LaPorte","PalmBeach","Smith","Knox","Pottawattamie","Wichita","Clatsop","Warrick","Cleveland","SanFrancisco","Cumberland","Washington","Middlesex","Delaware","Door","Madison","LosAngeles","Ventura","Horry","Suffolk"]
diag_codes = ["R000",	"I2510",	"R0602",	"I214",	"N390",	"N179",	"R509",	"N189",	"D62",	"E669",	"F419",	"R0989",	"Z0000",	"Z1231",	"R9431",	"Z90710",	"N281",	"C61",	"I252",	"E782",	"E119",	"N183",	"I350",	"I129",	"Z951",	"Z79899",	"E039",	"Z01818",	"R609",	"R262",	"J189",	"E7800",	"G4733",	"R338",	"F17210",	"I517",	"K5730",	"R079",	"Z8673",	"M549",	"I480",	"K449",	"I481",	"E785",	"J449",	"Z98890",	"Z86010",	"K635",	"R9439",	"M1990",	"I639",	"D649",	"Z8249",	"Z7901",	"R05",	"D72829",	"M48061",	"R112",	"J9811",	"Z7982",	"N400",	"J45909",	"R5383",	"R51",	"R918",	"R300",	"E559",	"E6601",	"M810",	"R001",	"F329",	"K219",	"R55",	"R7301",	"E876",	"R600",	"E1122",	"Z01810",	"R109",	"M545",	"R0600",	"I340",	"I10",	"E875",	"I4891",	"M109",	"R110",	"R531",	"R0789",	"I482",	"M4316",	"Z87891"]
revenue_centre_codes  = ["420",	"300",	"301",	"272",	"305",	"730",	"250",	"636",	"637",	"510",]
proc_codes = ["0W3P8ZZ",	"0DB98ZX",	"0D9670Z",	"0DJD8ZZ",	"027034Z",	"03CL0ZZ",	"5A1955Z",	"03CK0ZZ",	"0T768DZ",	"B54MZZA",	"5A1945Z",	"4A023N8",	"B24BYZZ",	"B2151ZZ",	"0SPC08Z",	"8E0W4CZ",	"0RG20A0",	"027F3ZZ",	"B3101ZZ",	"0SB20ZZ"]
states = ["Tennessee",	"Oregon",	"NewJersey",	"Iowa",	"Michigan",	"Texas",	"NewYork",	"Mississippi",	"NorthCarolina",	"Wisconsin",	"Pennsylvania",	"Washington",	"Oklahoma",	"SouthCarolina",	"Florida",	"Indiana",	"Ohio",	"California",	"Illinois",	"Minnesota",	"Massachusetts",	"Nebraska",	"Georgia"]

def process_input(inputData,scaler:MinMaxScaler):

      new_data = pd.DataFrame(inputData, index=[0])

      #Frequency based imputation
      if new_data['HCPCS_CODE'].item() not in hcpcs_codes:
           new_data['HCPCS_CODE'] = 'G8979'

      if new_data['COUNTY'].item() not in counties:
           new_data['COUNTY'] = 'COUNTY_X'

      if new_data['CODE'].item() not in diag_codes:
           new_data['CODE'] = 'CODE_X'

      if new_data['REVENUE_CENTER_CODE'].item() not in revenue_centre_codes:
           new_data['REVENUE_CENTER_CODE'] = 'REVENUE_CENTER_CODE_X'

      if new_data['PROCEDURE_CODE'].item() not in proc_codes:
           new_data['PROCEDURE_CODE'] = 'PROCEDURE_CODEX'

      if new_data['STATE'].item() not in states:
           new_data['STATE'] = 'STATE_X'
           


      dat_cols = [col for col in new_data.columns if 'DATE' in col]
      for col in dat_cols:
            new_data[col] = new_data[col].astype('datetime64[ns]')


      
      
      # 18	Expenses incurred prior to coverage. PROCEDURE_DATE < COVERAGE_START_DATE
      for i in range(len(new_data)):
            new_data.loc[i,'PROC_COVER_START_DIFF'] = (new_data.loc[i, "PROCEDURE_DATE"] - new_data.loc[i, "COVERAGE_START_DATE"]).days
      for i in range(len(new_data)):
            new_data.loc[i,'PROC_COVER_START_MISMATCH'] = new_data.loc[i, "PROCEDURE_DATE"] < new_data.loc[i, "COVERAGE_START_DATE"]

      #### 19	Expenses incurred after coverage terminated. PROCEDURE_DATE > COVERAGE_END_DATE
      for i in range(len(new_data)):
        new_data.loc[i,'PROC_COVER_END_DIFF'] = (new_data.loc[i, "PROCEDURE_DATE"] - new_data.loc[i, "COVERAGE_END_DATE"]).days
    
      for i in range(len(new_data)):
        new_data.loc[i,'PROC_COVER_END_MISMATCH'] = new_data.loc[i, "PROCEDURE_DATE"] < new_data.loc[i, "COVERAGE_START_DATE"]

      #10 The diagnosis is inconsistent with the patient's gender
      for i in range(len(new_data)):
            if  new_data.loc[i, "GENDER"] == 'female' and  new_data.loc[i, "CODE"] in code_gender:
                new_data.loc[i, 'CODE_AGE_MISMATCH'] = True
            else:
                 new_data.loc[i, 'CODE_AGE_MISMATCH'] = False

      ## 9	The diagnosis is inconsistent with the patient's age. 
      for i in range(len(new_data)):
                
                if  new_data.loc[i, "GENDER"] == 'female' and  new_data.loc[i, "CODE"] in code_age:
                     new_data.loc[i, 'CODE_GENDER_MISMATCH'] = True
                else:
                     new_data.loc[i, 'CODE_GENDER_MISMATCH'] = False

      #num_cols = ['Age', 'SERVICE_UNIT_QUANTITY', 'TOTAL_CHARGES']
      num_cols = ['Age', 'TOTAL_CHARGES', 'SERVICE_UNIT_QUANTITY', 'PROC_COVER_START_DIFF','PROC_COVER_END_DIFF']

      scaled = scaler.transform(new_data[num_cols])
      new_data['Age'] =  scaled[:,0]
      new_data['TOTAL_CHARGES'] =  scaled[:,1]
      new_data['SERVICE_UNIT_QUANTITY'] = scaled[:,2] 
      new_data['PROC_COVER_START_DIFF'] =  scaled[:,3]
      new_data['PROC_COVER_END_DIFF'] = scaled[:,4]   

      catcols = [k for k in new_data.columns if k not in num_cols]
      new_encoded = pd.get_dummies(new_data, columns=catcols)

      for col in encoded_data.columns:
            if col not in new_encoded.columns:
                  new_encoded[col] = 0

      drop_firsts = set(new_encoded.columns.tolist()).difference(encoded_data.columns.tolist())
      new_encoded.drop(columns=drop_firsts, inplace=True)

      return new_encoded[encoded_data.columns]

def predict(input_data):

    in_data = process_input(input_data,scaler)
    prediction_prob = model.predict_proba(in_data)
    index = prediction_prob[0].argmax()
    index_np = np.array(index)
    prediction  = label_encoder.inverse_transform(index_np.reshape(1,1)).item()
    score = prediction_prob[0,index].item()
    print(f'prediction :{prediction}  Score:{score:.2f}')

    return json.dumps({
        'Label':prediction,
        'Score':f'{score:.2f}'
    })

data = {
"CLAIM_START_DATE":"2017-10-17",
"BILL_TYPE_CODE":131,
"PLACE_OF_SERVICE_CODE":99,
"REVENUE_CENTER_CODE":"REVENUE_CENTER_CODE_X",
"SERVICE_UNIT_QUANTITY":1,
"TOTAL_CHARGES":250.42,
"HCPCS_CODE":"G8979",
"DISCHARGE_DISPOSITION_CODE":1,
"COVERAGE_START_DATE":"2016-01-01",
"COVERAGE_END_DATE":"2018-09-01",
"GENDER":"male",
"Age":"92",
"RACE":"white",
"STATE":"STATE_X",
"COUNTY":"COUNTY_X",
"CODE":"CODE_X",
"PRESENT_ON_ADMIT":0,
"DUAL_STATUS":2,
"MEDICARE_STATUS":10,
"ENCOUNTER_TYPE":"Other",
"ADMIT_SOURCE_CODE":4,
"ADMIT_TYPE_CODE":3,
"MS_DRG":378,
"PROCEDURE_DATE":"2015-10-16",
"PAYERS":"Cigna",
"PROCEDURE_CODE":"PROCEDURE_CODEX"
}

""" if __name__ =='__main__':
    data = json.loads(sys.argv[1])
    print(predict(data)) """
predict(data)


#input_data = "{\"CLAIM_START_DATE\":\"2017-10-17\",\"BILL_TYPE_CODE\":131,\"PLACE_OF_SERVICE_CODE\":99,\"REVENUE_CENTER_CODE\":\"REVENUE_CENTER_CODE_X\",\"SERVICE_UNIT_QUANTITY\":1,\"TOTAL_CHARGES\":250.42,\"HCPCS_CODE\":\"G8979\",\"DISCHARGE_DISPOSITION_CODE\":1,\"COVERAGE_START_DATE\":\"2016-01-01\",\"COVERAGE_END_DATE\":\"2018-09-01\",\"GENDER\":\"male\",\"Age\":\"92\",\"RACE\":\"white\",\"STATE\":\"STATE_X\",\"COUNTY\":\"COUNTY_X\",\"CODE\":\"CODE_X\",\"PRESENT_ON_ADMIT\":0,\"DUAL_STATUS\":2,\"MEDICARE_STATUS\":10,\"ENCOUNTER_TYPE\":\"Other\",\"ADMIT_SOURCE_CODE\":4,\"ADMIT_TYPE_CODE\":3,\"MS_DRG\":378,\"PROCEDURE_DATE\":\"2015-10-16\",\"PAYERS\":\"Cigna\",\"PROCEDURE_CODE\":\"PROCEDURE_CODEX\"}"