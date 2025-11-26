def init():
        
        gen=[]
        taken=set()
        for i in range(len(boxes)):
            while True:
                temp=random.randint(0,len(boxes)-1)
                if temp not in taken:
                    taken.add(temp)
                    break
            gen.append(boxes[temp])
        return gen