from sbyte import *
import math
def find_m(n):
    m=int(math.log2(n+1))

    if m+n+1>(1<<m):
        m+=1

    return m


def hamming_code(data):
    m=find_m(data.atomic_len())

    pad=math.ceil(m/8)


    encrypted_data=hex_to_bytes('00'*(pad+len(data)))

    parity=hex_to_bytes('00'*pad)

    parity_indices=set([1<<i for i in range(m)])
    # print(parity_indices)
    idx=1
    i=(pad+len(data))*8-1


    for byte in reversed(data):
        for bit in reversed(byte):
            while idx in parity_indices:
                idx+=1
                i-=1
            encrypted_data.set_atomic(i,bit)
            if bit==1:
                parity=parity^dec_to_bytes(idx)
            i-=1
            idx+=1

    for j in parity_indices:
        encrypted_data.set_atomic((pad+len(data))*8-j,parity.get_atomic(parity.atomic_len()-1-int(math.log2(j))))
        # print((pad+len(data))*8-j)

    return encrypted_data

print(ascii_to_bytes('hello'))
print(hamming_code(ascii_to_bytes('hello')))
#output: 00000000 01100110