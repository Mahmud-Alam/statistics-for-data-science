import math as m

# Function for exact Probability
def poisson_exact_probability(meu, x):
    prob = round((m.exp(-meu)*meu**x)/m.factorial(x),5)
    return prob

# Function for AT Least Probability (greater than or equal to the value)
def poisson_upperBound_at_least_probability(meu, x, n):
    prob = 0
    for i in range(int(x),int(n+1)):
        prob+=((m.exp(-meu)*meu**i)/m.factorial(i))
    return prob

# Function for AT Most Probability (Till the value)
def poisson_at_most_probability(meu, x):
    prob = 0
    for i in range(0,int(x+1)):
        value = round((m.exp(-meu)*meu**i)/m.factorial(i),5)
        prob+=value
        print(i,' prob: ',value)
    return prob

def poisson_at_least_probability(meu, x):
    prob = 0
    for i in range(0,int(x)):
        value = round((m.exp(-meu)*meu**i)/m.factorial(i),5)
        prob+=value
        print(i,' prob: ',value)
    return 1-prob
    

#%%

meu = float(input('Arrival rate - Meu (m) : '))
x = float(input("No of times event happened (x) : "))


#%%

# Code for Exact Probability
prob = poisson_exact_probability(meu, x)
print('\nExact Probability :', prob)

#%%

# Code for AT Least Probability (greater than or equal to the value)
prob = poisson_at_least_probability(meu, x)
print('\nAt Least Probability :',prob)

#%%

# Code for AT Most Probability (Till the value)
prob = poisson_at_most_probability(meu, x)
print('\nAt Most Probability :',prob)

#%%


n = float(input("Top bound (n) : "))
prob = poisson_upperBound_at_least_probability(meu, x, n)
print('\nAt Least Probability :',prob)