# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)

p_a = len(df[df['fico']>700])/len(df)
print("Probability of event that fico credit score is greater than 700 : ",p_a)

p_b = len(df[df['purpose']=='debt_consolidation'])/len(df)
print("Probability of event that purpose == 'debt_consolation' : ",p_b)

df1 = df[df['purpose']=='debt_consolidation']
p_a_b = len(df1[df1['purpose']=='debt_consolidation'])/len(df1[df1['fico']>700])
print("Probability of event for the event purpose == 'debt_consolidation' given 'fico' credit score is greater than 700 : ",p_a_b)

result = p_a_b==p_a
print(result)

# code ends here


# --------------
# code starts here

prob_lp = len(df[df['paid.back.loan']=='Yes'])/len(df)
print(prob_lp)

prob_cs = len(df[df['credit.policy']=='Yes'])/len(df)
print(prob_cs)

new_df = df[df['paid.back.loan']=='Yes']

prob_pd_cs = len(new_df[new_df['credit.policy']=='Yes'])/len(new_df[new_df['paid.back.loan']=='Yes'])
print(prob_pd_cs)

bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here

df['purpose'].value_counts(normalize=True).plot(kind='bar')
plt.title("Probability Distribution of Purpose")
plt.ylabel("Probability")
plt.xlabel("Number of Purpose")
plt.show()

df1 = df[df['paid.back.loan']=='No']

df1['purpose'].value_counts(normalize=True).plot(kind='bar')
plt.title("Probability Distribution of Purpose")
plt.ylabel("Probability")
plt.xlabel("Number of Purpose")
plt.show()

# code ends here


# --------------
# code starts here

inst_median = np.median(df['installment'])

inst_mean = np.mean(df['installment'])

df['installment'].plot(kind='hist', bins=50)
plt.axvline(x=inst_median,color='r')
plt.axvline(x=inst_mean,color='g')
plt.show()

df['log.annual.inc'].plot(kind='hist', bins=50)
plt.show()

# code ends here


