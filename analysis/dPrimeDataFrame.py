# Calculates number of trials, hit rate, and failure rate for each observer, difficulty level,
import pandas as pd
import os

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
HallData = pd.read_csv('HalloweenSubjectData.csv')

# Get variables
subjects = HallData.subject.unique()
sa = HallData['SA[seconds]'].unique()
difficulty = HallData.Difficulty.unique()

def makeDf():

    subVals = []
    idVals = []
    diffVals = []
    saVals =[]
    choiceVals = []
    numTrialsPre = []
    numTrialsPost = []

    hitValsPre = []
    hitValsPost = []

    failureValsPre = []
    failureValsPost = []

    # hitRatePreVals = []
    # hitRatePostVals = []
    #
    # failureRatePreVals = []
    # failureRatePostVals = []

    for sub in subjects:
        subData = HallData.loc[HallData.subject == sub]
        dates = subData.date.unique()

        for diff in difficulty:
            for item in sa:

                subDataPre = subData.loc[
                    (subData.date == dates[0]) | (subData.date == dates [1]) | (
                            subData.date == dates[2]) | (subData.date == dates[3]) | (
                            subData.date == dates[4]) | (subData.date == dates[5])]

                subDataPost = subData.loc[
                    (subData.date == dates[-6]) | (subData.date == dates[-5]) | (
                            subData.date == dates[-4]) | (subData.date == dates[-3]) | (
                            subData.date == dates[-2]) | (subData.date == dates[-1])]

                preNumTrials = len(
                    subDataPre.date[(subDataPre.Difficulty == diff) & (subDataPre['SA[seconds]'] == item)])

                postNumTrials = len(
                    subDataPost.date[(subDataPost.Difficulty == diff) & (subDataPost['SA[seconds]'] == item)])

                preHitTotal = len(subDataPre[
                                        (subDataPre.Difficulty == diff) & (subDataPre['SA[seconds]'] == item) & (
                                                subDataPre['ghost[bool]'] == True)])

                postHitTotal = len(subDataPost[
                                        (subDataPost.Difficulty == diff) & (subDataPost['SA[seconds]'] == item) & (
                                                subDataPost['ghost[bool]'] == True)])

                preFailureTotal = len(subDataPre[
                                        (subDataPre.Difficulty == diff) & (subDataPre['SA[seconds]'] == item) & (
                                                subDataPre['ghost[bool]'] == False)])

                postFailureTotal = len(subDataPost[
                                        (subDataPost.Difficulty == diff) & (subDataPost['SA[seconds]'] == item) & (
                                                subDataPost['ghost[bool]'] == False)])

                if sub == 'by':
                    subId = 'AA1'
                elif sub == 'co':
                    subId = 'AA2'
                elif sub == 'ky':
                    subId = 'AA3'
                elif sub == 'mb':
                    subId = 'AA4'
                elif sub == 'mg':
                    subId = 'AMS1'
                elif sub == 'bb':
                    subId = 'AS1'
                elif sub == 'et':
                    subId = 'AS2'
                elif sub == 'kp':
                    subId = 'AS3'
                elif sub == 'tp':
                    subId = 'AS4'
                elif sub == 'jz':
                    subId = 'ASW1'
                elif sub == 'ah':
                    subId = 'N1'
                elif sub == 'aj':
                    subId = 'N2'
                elif sub == 'dd':
                    subId = 'N3'
                elif sub == 'dl':
                    subId = 'N4'
                elif sub == 'ez':
                    subId = 'N5'
                elif sub == 'it':
                    subId = 'N6'
                elif sub == 'll':
                    subId = 'N7'
                elif sub == 'sh':
                    subId = 'N8'
                elif sub == 'sm':
                    subId = 'N9'
                elif sub == 'sr':
                    subId = 'N10'

                if item == 1000:
                    choice = 2
                elif item == 800:
                    choice = 3
                elif item == 600:
                    choice = 3
                elif item == 400:
                    choice = 4
                elif item == 300:
                    choice = 4
                elif item == 200:
                    choice = 4

                subVals.append(sub)
                idVals.append(subId)
                diffVals.append(diff)
                saVals.append(item)
                choiceVals.append(choice)
                numTrialsPre.append(preNumTrials)
                numTrialsPost.append(postNumTrials)

                hitValsPre.append(preHitTotal)
                hitValsPost.append(postHitTotal)
                failureValsPre.append(preFailureTotal)
                failureValsPost.append(postFailureTotal)

    return subVals, idVals, diffVals, saVals, choiceVals, numTrialsPre, numTrialsPost, hitValsPre, hitValsPost, failureValsPre, failureValsPost

subject_list, id_list, diff_list, sa_list, choice_list, trialPre_list, trialPost_list, hitPre_list, hitPost_list, failurePre_list, failurePost_list = makeDf()
d = {'subject': subject_list, 'id': id_list, 'level': diff_list, 'sa': sa_list, 'choices':choice_list, 'totalTrials_Pre': trialPre_list, 'totalHits_Pre': hitPre_list, 'totalFailures_Pre': failurePre_list, 'totalTrials_Post': trialPost_list, 'hitsPost': hitPost_list, 'failuresPost': failurePost_list}
df = pd.DataFrame(d)

df.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\d_prime_data.csv', index=False)
