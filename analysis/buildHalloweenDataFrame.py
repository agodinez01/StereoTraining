import pandas as pd
import os

#Run this if you have to create a new dataframe.
main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
HallData = pd.read_csv('HalloweenSubjectData.csv')

HallData.rename(columns= {"ghost[bool]": "ghost", "SA[seconds]": "SA", "cyclops[bool]": "cyclops", "distance[meters]": "distance", "Difficulty": "difficulty"}, inplace=True)

sa = HallData.SA.unique()
sa = [1000, 800, 600, 400]
diff = HallData.difficulty.unique()
groups = HallData.group.unique()
#subjects = HallData.subject.unique()
subjects = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'kp', 'ky', 'mb', 'tp', 'jz', 'mg'}

def getVals():
    #groupVals = []
    saVals = []
    totalTrue = []
    subVals = []
    conditionVals = []
    dateVals = []

    for sub in subjects:
        dates = []

        subData = HallData.loc[HallData.subject == sub]
        dates = subData.date.unique()
        for date in dates:

            for item in sa:

                a = len(HallData[(HallData.subject == sub) & (HallData.SA == item) & (HallData.ghost == True)])
                b = len(HallData[(HallData.subject == sub) & (HallData.SA == item)])

                if b == 0:
                    ratioT == 'nan'
                else:
                    ratioT = len(HallData[(HallData.subject == sub) & (HallData.SA == item) & (HallData.ghost == True)])/ \
                            len(HallData[(HallData.subject == sub) & (HallData.SA == item)])

                if sub == 'by' or sub == 'co' or sub =='ky' or sub == 'mb':
                    conditionVal = 'anisometropia'
                elif sub == 'bb' or sub == 'et' or sub == 'kp' or sub == 'tp':
                    conditionVal = 'strabismus'
                elif sub == 'ah' or sub == 'aj' or sub == 'dd' or sub == 'dl' or sub == 'ez' or sub == 'it' or sub == 'll' or sub == 'sh' or sub == 'sm' or sub == 'sr':
                    conditionVal = 'binocular'
                elif sub == 'mg' or sub == 'jz':
                    conditionVal = 'stereo-weak'

                subVals.append(sub)
                saVals.append(item)
                totalTrue.append(ratioT)
                conditionVals.append(conditionVal)
                dateVals.append(date)

    return subVals, saVals, totalTrue, conditionVals, dateVals

subVals, saVals, totalTrue, conditionVals, dateVals = getVals()
df = {'subject':subVals, 'condition':conditionVals, 'date':dateVals , 'sa': saVals, 'ratio': totalTrue}
dataFrame = pd.DataFrame(df)

dataFrame.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\HallHitRate.csv', index=False)

