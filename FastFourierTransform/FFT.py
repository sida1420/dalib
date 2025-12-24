import math
def FFT(function):
    if len(function)==1:
        return function

    n=len(function)
    
    oF=function[1::2]
    oA=FFT(oF)

    eF=function[0::2]
    eA=FFT(eF)

    
    theta=-2*math.pi/n





    ans=[0]*n
 
    for i in range(n//2):
        rea=math.cos(theta*i)
        ima=math.sin(theta*i)

        w=complex(rea,ima)

        oA[i]*=w
        
        ans[i]=eA[i]+oA[i]

        ans[i+n//2]=eA[i]-oA[i]
    return ans

        

print(FFT([0,1,2,3]))



