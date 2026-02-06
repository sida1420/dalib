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
def IFFT(function):
    if len(function)==1:
        return function

    n=len(function)
    
    oF=function[1::2]
    oA=IFFT(oF)

    eF=function[0::2]
    eA=IFFT(eF)

    
    theta=2*math.pi/n


    ans=[0]*n
 
    for i in range(n//2):
        rea=math.cos(theta*i)
        ima=math.sin(theta*i)

        w=complex(rea,ima)

        oA[i]*=w
        
        ans[i]=eA[i]+oA[i]

        ans[i+n//2]=eA[i]-oA[i]
    return ans

        


def mul(coeffs1, coeffs2):
    n=len(coeffs1)+len(coeffs2)-1

    rn=2**math.ceil(math.log2(n))

    discretePoints1=FFT(coeffs1+[0]*(rn-len(coeffs1)))
    discretePoints2=FFT(coeffs2+[0]*(rn-len(coeffs2)))
    print(discretePoints1)
    print(discretePoints2)

    productPoints=[]
    for i in range(0,rn):
        productPoints.append(discretePoints1[i]*discretePoints2[i])

                                   

    result=IFFT(productPoints)

    return [a.real/rn for a in result]

print(mul([2,1],[3,2]))

    
