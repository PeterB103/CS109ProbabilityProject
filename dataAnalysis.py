import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
# https://seaborn.pydata.org/generated/seaborn.histplot.html

TEAM_A_FILE = "1575_Auton - 1575A.csv"
TEAM_K_FILE = "1575_Auton - 1575K.csv"
A = "1575A"
K = "1575K"

#number of times to bootstrap
N_BOOTSTRAP = 10000

# Number of trials of robot in each bootstrap
sample_size = 22

def readData(fileName):
    filePath = fileName
    data = pd.read_csv(filePath)
    #points
    pointsScored = data['Points Scored']
    averagePoints = pointsScored.mean()
    variancePoints = pointsScored.var()

    #battery charge
    batteryCharge = data["Battery Percentage"]

    return data, pointsScored, averagePoints, batteryCharge, variancePoints

# Initial Histogram for Base Data
def scoreFrequency(teamName, pointsScored, averagePoints, variancePoints):
    plt.figure(figsize=(10, 6))
    sns.histplot(pointsScored, kde=True, bins=range(0, pointsScored.max() + 1, 5), discrete=True)
    plt.title(f'Distribution of Points Scored for {teamName}')
    plt.xlabel('Points Scored')
    plt.ylabel('Frequency')
    plt.text(x=0.05, y=13, s=f'Average Points: {averagePoints:.2f}\n Variance: {variancePoints:.2f}', fontsize = 12, bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8), ))
    plt.show()

#Plotting the battery to points correlation
def scatterBatteryToPoints(teamName, data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Points Scored', y='Battery Percentage', data=data)

    # Adding titles and labels
    plt.title(f'Points Scored vs Battery Percentage for {teamName}')
    plt.xlabel('Points Scored')
    plt.ylabel('Battery Percentage')
    plt.show()

#Plotting the number of times we succeed when ran for X trials
def trialsSuccessLineGraph(successRates, teamName,variance_of_means, means_of_means):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 23), successRates, marker='o', linestyle='-')
    plt.title(f'Success Rates vs. Number of Trials for {teamName}')
    plt.xlabel('Number of Trials')
    plt.ylabel('Success Rate')
    plt.grid(True)

    stats_text =  (f'Variance of Bootstrap means: {variance_of_means:.2f}\n'
                  f'Means of Bootstrap Means: {means_of_means:.2f}')
    
    plt.text(x=0.95, y=0.05, s=stats_text, fontsize=12, 
            bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)),
            horizontalalignment='right', verticalalignment='bottom', 
            transform=plt.gca().transAxes)    
    plt.show()

#Used to get the means through bootstrapping
def bootstrap_mean(pointsScored):
    bootstrap_means = []
    for _ in range(N_BOOTSTRAP):
        bootstrap_sample = np.random.choice(pointsScored, size=sample_size, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    return bootstrap_means

def plot_drawn_distribution(teamName, all_drawn_values):
    plt.figure(figsize=(10, 6))
    plt.hist(all_drawn_values, bins=24, density=True, alpha=0.6, color='g')
    plt.title(f'Distribution of Drawn Values for {teamName}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()
    
def simulate_trials(teamName, pointsScored, mean, trueAverage, improvement=False):
    counter_success = 0
    stdDev = np.std(pointsScored)
    #basing our minimum desired score based off of our average from our 22 trials
    minimumDesireScore = trueAverage
    singleTrialProb = 0
    maxScore = pointsScored.max()

    successRates = [] #rates for #trials 1-22
    all_drawn_values = []
    #simulate which number of trials is best
    for nTrials in range(1,23):
        counter_success = 0
        for _ in range(N_BOOTSTRAP):
            simulated_trial_score = np.random.normal(mean, stdDev, nTrials)
            if improvement:
                for i in range(len(simulated_trial_score)):
                        while simulated_trial_score[i] < 0 or simulated_trial_score[i] > maxScore: #keep redrawing
                            simulated_trial_score[i] = np.random.normal(mean, stdDev, 1)

            all_drawn_values.extend(simulated_trial_score)
            if np.mean(simulated_trial_score) >= (minimumDesireScore - stdDev):
                counter_success += 1
        successRates.append(counter_success / N_BOOTSTRAP)

        #get the probability of my robot scoring above its true mean off of one trial
        if nTrials == 1:
            singleTrialProb = counter_success / N_BOOTSTRAP

    # Plotting the distribution of all drawn values
    plot_drawn_distribution(teamName, all_drawn_values)
    

    return successRates, singleTrialProb

def simulate_bernoulli(p):
    # We did this one for you!
    if np.random.uniform(0, 1) < p:
        return 1
    return 0

def trialsToSuccessInNRow(singleTrialProb, TargetSuccess):
    nTrials = 0
    currentSuccessInARow = 0
    while True:
        nTrials += 1

        #we've succeeded increment
        if(simulate_bernoulli(singleTrialProb) == 1):
            currentSuccessInARow += 1

        #Reset the number of successes we've had so far
        else:
            currentSuccessInARow = 0

        if currentSuccessInARow == TargetSuccess:
            #it took nTrials to reach TargetSuccess
            return nTrials
        
def desiredSuccess(singleTrialProb):
    nSuccessesToNTrials = {}
    for nSuccess in range(2,6):
        trials = []
        for _ in range(N_BOOTSTRAP):
            trials.append(trialsToSuccessInNRow(singleTrialProb, nSuccess))
        nSuccessesToNTrials[nSuccess] = np.mean(trials)

    # Prepare data for histogram
    trials = list(nSuccessesToNTrials.values())

    # Prepare data for plotting
    successes = list(nSuccessesToNTrials.keys())
    trials = list(nSuccessesToNTrials.values())

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(successes, trials, marker='o', color='blue')
    plt.title('Number of Trials to Achieve Consecutive Successes')
    plt.xlabel('Number of Consecutive Successes')
    plt.ylabel('Number of Trials')
    plt.xticks(successes)  # Ensure all keys are displayed on x-axis
    plt.grid(True)  # Adding grid for better readability
    plt.show()

def conditional_prob(pointsScored, batteryCharge, desiredX, desiredCharge, flipped=False):
    greaterThanX = 0
    greaterThanCharge = 0
    total_scores = len(pointsScored)
    #Event A
    for score in pointsScored:
        #keep track of all instances that is above our desired score
        if score >= desiredX:
            greaterThanX += 1
    probability_A = greaterThanX / total_scores
    print(f'(A:{probability_A}') 

    #EventB: Accounts for both sides of the charge
    for charge in batteryCharge:
        if flipped:
            if charge <= desiredCharge:
                greaterThanCharge += 1
        else:
            if charge >= desiredCharge:
                greaterThanCharge += 1
    probability_B = greaterThanCharge / total_scores
    print(f'(B:{probability_B}') 


    #P(A and B)
    greaterThanX_and_greaterThanCharge = 0
    if flipped:
        for i in range(total_scores):
            if pointsScored[i] >= desiredX and batteryCharge[i] <= desiredCharge:
                greaterThanX_and_greaterThanCharge += 1
    else:
        for i in range(total_scores):
            if pointsScored[i] >= desiredX and batteryCharge[i] >= desiredCharge:
                greaterThanX_and_greaterThanCharge += 1
    probability_A_and_B = greaterThanX_and_greaterThanCharge / total_scores
    print(probability_A_and_B)

    #Conditional Probability
    #P(A|B) = P(A and B) / P(B)
    if probability_B <= 0:
        probability_A_given_conditional_B = 0
    else:
        probability_A_given_conditional_B = probability_A_and_B / probability_B
    return probability_A_given_conditional_B

def team_analysis(team_name, team_file, desiredX, desiredCharge):
    # Read data, og score, og battery charge/plot
    data, pointsScored, averagePoints, batteryCharge, variancePoints = readData(team_file)
    scoreFrequency(team_name, pointsScored, averagePoints, variancePoints)
    scatterBatteryToPoints(team_name, data)

    #Conditional Probability
    print(f'{team_name}')
    #greater than 80 battery
    probability_A_given_conditional_B = conditional_prob(pointsScored, batteryCharge, desiredX, desiredCharge)
    print(f'P(Points | Battery > 80) equivalent to Conditional of A given B:{probability_A_given_conditional_B}\n')
    #Less than 80 battery
    probability_A_given_conditional_B = conditional_prob(pointsScored, batteryCharge, desiredX, desiredCharge, flipped=True)
    print(f'P(Points | Battery < 80) equivalent to Conditional of A given B:{probability_A_given_conditional_B}\n')

    #Bootstrapping analysis
    bootstrap_means = bootstrap_mean(pointsScored)
    variance_of_means = np.var(bootstrap_means)
    means_of_means = np.mean(bootstrap_means)

    # Optimal Trial Count and success rate analysis and gathers success rate for running nTrials for 1000 bootsrapping times
    trueAverage = averagePoints
    successRates, singleTrialProb = simulate_trials(team_name, pointsScored, averagePoints, trueAverage)
    trialsSuccessLineGraph(successRates, team_name, variance_of_means, means_of_means)
    desiredSuccess(singleTrialProb)

    # Running with improved normal drawing
    successRates_improved, singleTrialProb_improved = simulate_trials(team_name, pointsScored, averagePoints, trueAverage, improvement=True)
    trialsSuccessLineGraph(successRates_improved, team_name, variance_of_means, means_of_means)
    desiredSuccess(singleTrialProb_improved)

def main():
    sns.set(style="whitegrid")
    # desiredX = 10
    # desiredCharge = 80
    # team_analysis(A, TEAM_A_FILE, desiredX, desiredCharge )

    desiredX = 3
    desiredCharge = 80
    team_analysis(K, TEAM_K_FILE, desiredX, desiredCharge)

main()