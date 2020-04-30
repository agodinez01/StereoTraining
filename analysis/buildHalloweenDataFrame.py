import pandas as pd
import os

#Run this if you have to create a new dataframe.
main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
HallData = pd.read_csv('HalloweenSubjectData.csv')

HallData.rename(columns= {"ghost[bool]": "ghost", "SA[seconds]": "SA", "cyclops[bool]": "cyclops", "distance[meters]": "distance", "Difficulty": "difficulty"}, inplace=True)

sa = [1000, 800, 600, 400]
diff = HallData.difficulty.unique()
groups = HallData.group.unique()
subjects = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'kp', 'ky', 'mb', 'tp', 'jz', 'mg'}

def getVals():
    #groupVals = []
    saVals = []
    totalTrue = []
    subVals = []
    conditionVals = []
    dateVals = []
    diffVals = []

    for sub in subjects:
        subData = []

        subData = HallData.loc[HallData.subject == sub]
        dates = []
        dates = subData.date.unique()
        for date in dates:

            for item in sa:
                for d in diff:

                    a = len(subData[(subData.date == date) & (subData.SA == item) & (subData.ghost == True) & (subData.difficulty == d)])
                    b = len(subData[(subData.date == date) & (subData.subject == sub) & (subData.SA == item) & (subData.difficulty == d)])

                    # a = len(HallData[(HallData.subject == sub) & (HallData.SA == item) & (HallData.ghost == True) & (HallData.difficulty == d)])
                    # b = len(HallData[(HallData.subject == sub) & (HallData.SA == item) & (HallData.difficulty == d)])
                    #
                    if b == 0:
                        ratioT = 'nan'
                    else:
                        ratioT = a/b
                        #ratioT = len(subData[(subData.subject == sub) & (subData.SA == item) & (subData.ghost == True) & (subData.difficulty == d)])/ \
                         #       len(subData[(subData.subject == sub) & (subData.SA == item) & (subData.difficulty == d)])

                    if sub == 'by' or sub == 'co' or sub =='ky' or sub == 'mb':
                        conditionVal = 'anisometropia'
                    elif sub == 'bb' or sub == 'et' or sub == 'kp' or sub == 'tp':
                        conditionVal = 'strabismus'
                    elif sub == 'ah' or sub == 'aj' or sub == 'dd' or sub == 'dl' or sub == 'ez' or sub == 'it' or sub == 'll' or sub == 'sh' or sub == 'sm' or sub == 'sr':
                        conditionVal = 'binocular'
                    elif sub == 'mg' or sub == 'jz':
                        conditionVal = 'stereo-weak'

                    if sub == 'by':
                        subV = 'AA1'
                    elif sub == 'co':
                        subV = 'AA2'
                    elif sub == 'ky':
                        subV = 'AA3'
                    elif sub == 'mb':
                        subV = 'AA4'
                    elif sub == 'mg':
                        subV = 'AMS1'
                    elif sub == 'bb':
                        subV = 'AS1'
                    elif sub == 'et':
                        subV= 'AS2'
                    elif sub == 'kp':
                        subV = 'AS3'
                    elif sub == 'tp':
                        subV = 'AS4'
                    elif sub == 'jz':
                        subV = 'ASW1'
                    elif sub == 'ah':
                        subV = 'N1'
                    elif sub == 'aj':
                        subV = 'N2'
                    elif sub == 'dd':
                        subV = 'N3'
                    elif sub == 'dl':

                        subV = 'N4'
                    elif sub == 'ez':
                        subV = 'N5'
                    elif sub == 'it':
                        subV = 'N6'
                    elif sub == 'll':
                        subV = 'N7'
                    elif sub == 'sh':
                        subV = 'N8'
                    elif sub == 'sm':
                        subV = 'N9'
                    elif sub == 'sr':
                        subV = 'N10'

                    subVals.append(subV)
                    saVals.append(item)
                    totalTrue.append(ratioT)
                    conditionVals.append(conditionVal)
                    dateVals.append(date)
                    diffVals.append(d)

    return subVals, saVals, totalTrue, conditionVals, dateVals, diffVals

subVals, saVals, totalTrue, conditionVals, dateVals, diffVals = getVals()
df = {'subject':subVals, 'difficulty':diffVals, 'condition':conditionVals, 'date':dateVals , 'sa': saVals, 'ratio': totalTrue}
dataFrame = pd.DataFrame(df)

dataFrame.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\HallHitRate4.csv', index=False)

