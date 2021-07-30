# My own mean function

def own_mean(a):
    SUM = 0
    count = 0
    for i in a:
        SUM+=i
        count+=1
    
    mean = SUM/count
    return mean

### END MEAN FUNCTION