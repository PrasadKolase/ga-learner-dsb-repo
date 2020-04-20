# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data = np.genfromtxt(path, delimiter=",", skip_header=1)

census = np.concatenate((data,new_record))


# --------------
#Code starts here
age = census[:,0]
#print(age)
max_age = np.max(age)
#print(max_age)
min_age = np.min(age)
#print(min_age)
age_mean = np.mean(age)
#print(age_mean)
age_std = np.std(age)
#print(age_std)

# 90 17 38.06 13.34


# --------------
#Code starts here
race_0 = census[census[:,2]==0]
race_1 = census[census[:,2]==1]
race_2 = census[census[:,2]==2]
race_3 = census[census[:,2]==3]
race_4 = census[census[:,2]==4]
a = [race_0,race_1,race_2,race_3,race_4]

len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
b = [len_0,len_1,len_2,len_3,len_4]

minority_race = b.index(min(len_0,len_1,len_2,len_3,len_4))
print(minority_race)



# --------------
#Code starts here
senior_citizens = census[census[:,0]>60]
print(senior_citizens)

working_hours_sum = sum(census[:,6][age>60])

senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum/senior_citizens_len

print(avg_working_hours)


# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]

avg_pay_high = np.mean(census[census[:,1]>10][:,7])
avg_pay_low = np.mean(census[census[:,1]<=10][:,7])

print("Highly Educated Peoples : ",avg_pay_high)

print("Low Educated Peoples : ",avg_pay_low)
