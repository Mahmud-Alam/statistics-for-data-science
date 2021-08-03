import math as m

# Function for exact Probability
def exact_probability(n,x,p):
    prob = (m.factorial(n)/(m.factorial(x)*m.factorial(n-x)))*(p**x)*((1-p)**(n-x))
    return prob

# Function for AT Least Probability (greater than or equal to the value)
def at_least_probability(n,x,p):
    prob = 0
    for i in range(int(x),int(n+1)):
        prob+=((m.factorial(n)/(m.factorial(i)*m.factorial(n-i)))*(p**i)*((1-p)**(n-i)))
    return prob

# Function for AT Most Probability (Till the value)
def at_most_probability(n,x,p):
    prob = 0
    for i in range(int(x+1),int(n+1)):
        prob+=(m.factorial(n)/(m.factorial(i)*m.factorial(n-i)))*(p**i)*((1-p)**(n-i))
    final_prob = 1-prob
    return final_prob
    

#%%

n = float(input('How many times (n) : '))
x = float(input("What amount's probability (x) : "))
p = float(input('Probability in % value (p) : '))/100

mean = n*p
variance = n*p*(1-p)

print('\nMean :',round(mean,6))
print('Variance :',round(variance,6))

#%%

# Code for Exact Probability
prob = exact_probability(n, x, p)
print('\nExact Probability :',round(prob,6))

#%%

# Code for AT Least Probability (greater than or equal to the value)
prob = at_least_probability(n, x, p)
print('\nAt Least Probability :',round(prob,6))

#%%

# Code for AT Most Probability (Till the value)
prob = at_most_probability(n, x, p)
print('\nAt Most Probability :',round(prob,6))


