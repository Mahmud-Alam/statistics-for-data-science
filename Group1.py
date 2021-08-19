import math


# MAX FUNCTION
def MAX(List):
    MAX=0
    for i in List:
        if i>MAX:
            MAX=i
    return MAX


# MIN FUNCTION
def MIN(List):
    MIN=99999
    for i in List:
        if i<MIN:
            MIN=i
    return MIN


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


# LENGTH FUNCTION
def LEN(List):
    n=0
    for i in List:
        n+=1    
    return n


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


# MEAN_ABSOLUTE_DEVIATION FUNCTION
def MEAN_ABSOLUTE_DEVIATION(List):
    xBar = MEAN(List)
    xMinusXbar = [abs(x-xBar) for x in List]
    total = SUM(xMinusXbar)
    n = LEN(List)
    return total/n


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

