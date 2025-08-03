# Deployment for adult income data
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import IsolationForest
import json
import matplotlib.pyplot as plt
import seaborn as sns

education_dict = {'HS-grad' : 9,
'Some-college' : 10,
'Bachelors' : 13,
'Masters' : 14,
'Assoc-voc' : 11,
'11th' : 7,
'Assoc-acdm' : 12,
'10th' : 6,
'7th-8th' : 4,
'Prof-school' : 15,
'9th' : 5,
'12th' : 8,
'Doctorate' : 16,
'5th-6th' : 3,
'1st-4th' : 2,
'Preschool' : 1}

# Page title
st.title('Income prediction')
st.markdown('This model predicts if the income of a person will be >50K or <=50K based on the input parameters. This model ' \
'uses the Tuned XGBoosting algorithm which appears to be the best based on training and testing the dataset on various ' \
'binary classification algorithms')

age = st.sidebar.slider("Age",min_value=17,max_value=90,value=30)

workclass = st.sidebar.selectbox("Workclass",['state-gov', 'self-emp-not-inc', 'private', 'local-gov', 'self-emp-inc', 'without-pay', 'never-worked'])

education = st.sidebar.selectbox("Education",['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad',
                                                  'Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'])

maritalstatus = st.sidebar.selectbox("Marital status",['never-married', 'married-civ-spouse', 'married-spouse-absent', 'separated', 'married-af-spouse','widowed'])

occupation = st.sidebar.selectbox("Occupation",['exec-managerial', 'handlers-cleaners','prof-specialty', 
        'other-service', 'sales', 'craft-repair','transport-moving', 'farming-fishing', 'machine-op-inspct'])

relationship = st.sidebar.selectbox("Relationship",['not-in-family',  'wife', 'own-child', 'unmarried',
       'other-relative'])

race = st.sidebar.selectbox("Race",['white', 'black', 'asian-pac-islander', 'amer-indian-eskimo','other'])

sex = st.sidebar.radio("Gender",['Male','Female'])

capitalgain = st.sidebar.number_input("Capital Gain")

capitalloss = st.sidebar.number_input("Capital Loss")

hours = st.sidebar.slider("Hours per week",min_value=0,max_value=90,value=40)

nativecountry = st.sidebar.selectbox("Native country",['united-states', 'cuba', 'other', 'india', 'mexico', 'puerto-rico',
       'germany', 'philippines', 'el-salvador', 'china'])

# read the column values from the json file
with open('col_names.json','r') as f:
    cols = json.load(f)

# fill the values with 0
df = pd.DataFrame([[0]*len(cols)], columns=cols)

# fill numeric values
df['age'] = age
df['capital-gain'] = capitalgain
df['capital-loss'] = capitalloss
df['hours-per-week'] = hours
df['education-num'] = education_dict[education]
if sex=='Male':
    df['sex'] = 1

# categorical values
col_prefix = ['workclass_','occupation_','marital-status_','relationship_','race_','native-country_']
col_values = [workclass,occupation,maritalstatus,relationship,race,nativecountry]

cols_to_update = [x + y for x, y in zip(col_prefix, col_values)]

for col in cols_to_update:
    df[col] = 1

# df is updated with input values. Start the prediction
# Load the data
basic_models = {
'LR_model': joblib.load("s_model_LR.pkl"),
'RF_model': joblib.load("s_model_RF.pkl"),
'SVM_model': joblib.load("s_model_SVM.pkl"),
'XGB_model': joblib.load("s_model_XGB.pkl"),
'KNN_model': joblib.load("s_model_KNN.pkl")
}

tuned_models = {
    'Tuned_LR': joblib.load("s_tuned_LR.pkl"),
    'Tuned_RF': joblib.load("s_tuned_RF.pkl"),
    'Tuned_SVM': joblib.load("s_tuned_SVM.pkl"),
    'Tuned_XGB': joblib.load("s_tuned_XGB.pkl"),
    'Tuned_KNN': joblib.load("s_tuned_KNN.pkl")
}
df_plots = pd.read_csv('Results_Standardized.csv')

fr =  [[] for _ in range(len(df_plots))]
tr =  [[] for _ in range(len(df_plots))]
scores =  [[] for _ in range(len(df_plots))]
cmatrix = [[] for _ in range(len(df_plots))]

#cmatrix = [[],[] for _ in range(len(df_plots))]
for i in range(len(df_plots)):
    #parsing false_ratio to decode the list from the string
    false_ratio = df_plots.loc[df_plots['Algorithms'] == df_plots['Algorithms'][i],'False_Ratio'].values[0]
    false_ratio = false_ratio.translate(str.maketrans("","","[],"))
    temp=false_ratio.split(' ')
    false_ratio = [float(x) for x in temp]
    fr[i].append(false_ratio)


    #parsing function to decode the list from the string
    true_ratio = df_plots.loc[df_plots['Algorithms'] == df_plots['Algorithms'][i],'True_Ratio'].values[0]
    true_ratio = true_ratio.translate(str.maketrans("","","[],"))
    temp=true_ratio.split(' ')
    true_ratio = [float(x) for x in temp]
    tr[i].append(true_ratio)

    #roc score values
    score = df_plots.loc[df_plots['Algorithms'] == df_plots['Algorithms'][i],'AUC_Score'].values[0]
    scores[i].append(score)

    # confusion matrix values
    matrix = df_plots.loc[df_plots['Algorithms'] == df_plots['Algorithms'][i],'Matrix'].values[0]
    matrix = matrix.translate(str.maketrans("","","[]\n,"))
    temp=matrix.split(' ')
    matrix = [int(x) for x in temp]
    #list1 = matrix[:2]
    #list2 = matrix[2:]
    #matrix = [list1,list2]
    cmatrix[i].append(matrix)
#end of for loop

model_dict = {
    'LogisticRegression' : 0,
    'RandomForest' : 1,
    'XGBoosting' : 2,
    'SVM' : 3,
    'KNN' : 4,
    'TunedRandomForest' : 5,
    'TunedLogisticRegression': 6,
    'TunedXGBoosting' : 7,
    'TunedSVM' : 8,
    'TunedKNN' : 9
}

model = st.selectbox("Select a model ",['LogisticRegression','RandomForest','XGBoosting','SVM','KNN','TunedRandomForest',
                        'TunedLogisticRegression','TunedXGBoosting','TunedSVM','TunedKNN'])
m_index = model_dict[model]

#display the analytics data
# Classification report
cr = df_plots.loc[df_plots['Algorithms'] == model,'Report'].values[0]
st.write('******** CLASSIFICATION_REPORT ******** \n',cr)

# Confusion matrix
matrix = cmatrix[m_index]
list1 = matrix[:2]
list2 = matrix[2:]
matrix = [list1,list2]
fig = plt.figure()
sns.heatmap(matrix, annot=True, fmt='d')
st.pyplot(fig)

# Roc curve
fig, ax = plt.subplots()
ax.plot(fr[m_index], tr[m_index], label=f"{model} (AUC = {scores[m_index]:.2f})")
ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
  
# Show in Streamlit
st.pyplot(fig)

if st.button("Predict"):
    model = tuned_models['Tuned_XGB']
    prediction = model.predict(df)      
    #st.write("Predicted class for the given input : {prediction[0]}")
    if prediction[0] == 0:
        st.write("Based on the given input, the income will be <=50K")
    else:
        st.write("Based on the given input, the income will be >50K")

probability = tuned_models['Tuned_XGB'].predict_proba(df)[0]
class_names = tuned_models['Tuned_XGB'].classes_

#st.write(f" Predicted Class: **{prediction[0]}**")
st.write("Predication Probability")
# Show probabilities in a table
prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probability
    }).sort_values(by="Probability", ascending=False)

st.bar_chart(prob_df.set_index("Class"))
