import math


#count
#number of non NA values
def COUNT (List):
    c=0    
    for i in List:
        if (i!=None):
            c= c+1
    return c
            

#describe
def DESCRIBE (List):
    des ={}
    mean = MEAN(List)
    std = STD(List)
    Min = MIN(List)
    q1 = QUANTILE(List,.25)
    q2 = QUANTILE(List,.50)
    q3 = QUANTILE(List,.75)
    maxi = MAX(List)
    des = {
        "mean: " : mean,
        "std: " : std,
        "min: " : Min,
        "25%: " : q1,
        "50%: " : q2,
        "75%: " : q3,
        "max: " : maxi,
        }
    return des


# MAX FUNCTION
def MAX(List):
    MAX=0
    for i in List:
        if i>MAX:
            MAX=i
    return MAX


#INDEX OF MAX
def ARGMAX(List):
    count=0
    maximum = MAX(List)
    for j in List:
        if(j!=maximum):
            count = count + 1
        else:
            return count


# MIN FUNCTION
def MIN(List):
    MIN=99999
    for i in List:
        if i<MIN:
            MIN=i
    return MIN


#INDEX OF MIN
def ARGMIN(List):
    count=0
    minimum = MIN(List)            
    for j in List:
        if(j!=minimum):
            count = count + 1
        else:
            return count


#index lebel at which minimum lies
def IDX_MIN (List):
    minimum = List[0]
    index = 0
    for i in range(LEN(List)):
        if(List[i] < minimum):
            minimum = List[i]
            index=i
    return index    


#index lebel at which maximum lies
def IDX_MAX (List):
    maximum = List[0]
    index = 0
    for i in range(LEN(List)):
        if(List[i] > maximum):
            maximum = List[i]
            index=i
    return index


def QUANTILE (List, per):
    new_list= sorted(List)
    n=LEN(List)
    Q=0
    if per>0 and per<1:
        q = per
        Q = int(q*(n+1))
        return new_list[Q]
    else:
        print("Out of range...")
        return None
        

def PERCENTILE(List,per):
    x=per/100
    return QUANTILE(List,x)        
                

def MEDIAN(List):
    mid = QUANTILE (List, 0.5)
    return mid


def sort_Manual(x):
    x = x.sort_values()
    x.reset_index(inplace=True, drop=True)
    return x


# LENGTH FUNCTION
def LEN(List):
    n=0
    for i in List:
        n+=1    
    return n


# Statistical 
#count
def count_Manual(x):
    count = 0
    for i in x:
        count = count+1
    return count


#returns scientific notation of int values
def scientific_Method(x): 
    if(x<1e10):
        return x
    sci = ""
    st = str(x)
    p=0
    for i in st:  
       sci+= i
       if p==0:
           sci +='.'
       if p==6:
           break
       p+=1
    sci+="e+"+str(LEN(str(x))-1)   
    return sci


#quantile
def Quantile(a,x):
    a = sort_Manual(a)
    total = count_Manual(a)
    index = a
    quan = int(total*x)
    for i in range(LEN(index)):
        if i==quan:
            return index[i]
        

#sum
def Sum(a):
    s=0
    for i in a:
        s+=i
    return s


# Mean
def Mean(a):
    s = Sum(a)
    n = LEN(a)
    return s/n;


# median
def Median(a):
    a = sort_Manual(a)
    n = LEN(a) 
    if n&1:
        return  a[n//2 + 1] 
    return (a[n//2] + a[n//2 + 1])/2


#MAD
def Mad(arr):
    n = LEN(arr)
    M = Mean(arr)
    x=0
    for i in range(n):
        x += abs(arr[i]-M)
    return x/n

  
def Prod(a):
    prod = 1
    for i in range(LEN(a)):
        prod *= int(a[i])
    return scientific_Method(prod)

# variance
def Var(arr):
    n = LEN(arr)
    s = 0
    for i in range(n):
        x = (arr[i] - Mean(arr))
        s +=  x*x
    return s/(n-1)


# standardDeviation
def Std(arr):
    return math.sqrt(Var(arr))


# SUM FUNCTION
def SUM(List):
    if type(List[0])==str:
        st=''
        for i in List:
            st+=i
        return st
    SUM=0
    for i in List:
        SUM+=i    
    return SUM




# MEAN FUNCTION
def MEAN(List):
    s = SUM(List)
    n = LEN(List)
    return s/n


# VARIANCE FUNCTION
def VAR(List):
    xBar = MEAN(List)
    xMinusXbar = [(x-xBar)**2 for x in List]
    total = SUM(xMinusXbar)
    n = LEN(List)
    return total/(n-1)


# STD FUNCTION
def STD(List):
    var = VAR(List)
    return var**.5


# SKEW FUNCTION
def SKEW(List):
    xBar = MEAN(List)
    xMinusXbar = [(x-xBar)**3 for x in List]
    total = SUM(xMinusXbar)
    n = LEN(List)
    std = STD(List)
    return total/((n-1)*std**3)


# KURT FUNCTION
def KURT(List):
    xBar = MEAN(List)
    xMinusXbar = [(x-xBar)**4 for x in List]
    total = SUM(xMinusXbar)
    n = LEN(List)
    std = STD(List)
    return total/((n-1)*(std**4))


# CUMSUM FUNCTION
def CUMSUM(List):
    cum=0
    arr=[]
    for i in List:
        cum+=i
        arr.append(cum)
    return arr


# CUMMIN FUNCTION
def CUMMIN(List):
    MIN=List[0]
    cumulative=[]
    for i in List:
        if i<MIN:
            MIN=i
        cumulative.append(MIN)
    return cumulative


# CUMMAX FUNCTION
def CUMMAX(List):
    MAX=List[0]
    cumulative=[]
    for i in List:
        if i>MAX:
            MAX=i
        cumulative.append(MAX)
    return cumulative


# CUMPROD FUNCTION
def CUMPROD(List):
    cum=1
    j=0
    product=[]
    for i in List:
        cum*=List[j]
        product.append(cum)
        j+=1
    return product


# DIFF FUNCTION
def DIFF(List):
    diff=math.nan
    difference=[]
    j=0
    for i in List:
        diff=(List[j]-diff)*1.0
        difference.append(diff)
        diff=List[j]
        j+=1
    return difference


# PCT_CHANGE FUNCTION
def PCT_CHANGE(List):
    pct=math.nan
    percentage=[]
    j=0
    for i in List:
        pct=round((List[j]/pct)-1.0,6)
        percentage.append(pct)
        pct=List[j]
        j+=1
    return percentage


# TRIMMED_MEAN FUNCTION
def TRIMMED_MEAN(List, P):
    List=List.sort_values(ascending=True)
    List=List.iloc[P:-P]
    List = List.reset_index(drop=True)
    s = SUM(List)
    n = LEN(List)
    return s/n


# WEIGHTED_MEAN FUNCTION
def WEIGHTED_MEAN(List, weight):
    li=[List[i]*weight[i] for i in range(0,LEN(List))]
    total=SUM(li)
    sumWeight=SUM(weight)
    return total/sumWeight


# WEIGHTED_MEDIAN FUNCTION
def WEIGHTED_MEDIAN(List, weight):
    mid=SUM(weight)/2
    cumuWeight=CUMSUM(weight)
    for i in range(0,LEN(List)):
        if(cumuWeight[i]>=mid):
            index=i
            break
    return List[index]


# MODE FUNCTION
def MODE(List):
    MAX=0
    for i in range(0,LEN(List)):
        frq=0
        for j in range(0,LEN(List)):
            if List[i]==List[j]:
                frq+=1
        if(frq>=MAX):
            MAX=frq
            index=i
    return List[index]


# DISPERSION FUNCTION
def DISPERSION(List):
    return MAX(List)-MIN(List)


# ZSCORE FUNCTION
def ZSCORE(List):
    m = MEAN(List)
    s = STD(List)
    ZScore=[]
    for x in List:
        val=round((x-m)/s,4)
        ZScore.append(val)
    return ZScore


# COVARIANCE FUNCTION
def COVARIANCE(list1, list2):
    mean1 = MEAN(list1)
    mean2 = MEAN(list2)
    n = LEN(list1)
    va = [(list1[i]-mean1)*(list2[i]-mean2) for i in range(0,LEN(list1))]
    total = SUM(va)
    return round(total/(n-1),4)


# CORRELATION FUNCTION
def CORRELATION(list1, list2):
    cov = COVARIANCE(list1, list2)
    std1 = STD(list1)
    std2 = STD(list2)
    return round(cov/(std1*std2),4)


#interquantile range
def INTERQUARTILE_RANGE(a):
    a = sort_Manual(a)
    return Quantile(a,0.75) - Quantile(a,0.25)


# Standard Error
def StandardError(a):
  se = Std(a)/math.sqrt(count_Manual(a))
  return se

#z-score
def zscore(a,b):
  z = (a-Mean(b))/Std(b)
  return z


#Confidence Interval
def ConfidenceInterval(a,b):
  left  = Mean(b)-zscore(a,b)*StandardError(b)
  right = Mean(b)+zscore(a,b)*StandardError(b)
  return str(left) + "," + str(right)

