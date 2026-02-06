
import Classes as cl
import queue
import math
#POV facing in
def innerAngle(p1, p2,  p3):
    theta=cl.orientation(p1.x,p1.y,p2.x,p2.y,p3.x,p3.y)

    return theta>0



class node:
    def __init__(self,i, pre, next):
        self.i=i
        self.prev=pre
        self.next=next



def earClipping(polygon):
    n=len(polygon.points)
    work=[node(i%n,(i-1)%n,(i+1)%n) for i in range(n)]
    task=queue.Queue()
    reflex=set()
    for i in range(n):
        if innerAngle(polygon.points[work[i].next],polygon.points[work[i].i],polygon.points[work[i].prev]):
            task.put(i)
        else:
            reflex.add(i)

    triangles=[]

    done=n

    while done>2:
        i=task.get_nowait()
        
        ob=cl.Triangle(points=[polygon.points[work[i].prev],polygon.points[work[i].i],polygon.points[work[i].next]],coeff=polygon.coeff)

        safe=True
        for j in reflex:
            if ob.pointIntersect(polygon.points[j]):
                safe=False
                break

        if not safe:
            task.put(i)
            continue


        triangles.append(ob)
        done-=1



        if work[i].prev in reflex and innerAngle(polygon.points[work[i].next],polygon.points[work[i].prev],polygon.points[work[work[i].prev].prev]):
            task.put(work[i].prev)
            reflex.remove(work[i].prev)
        
        if work[i].next in reflex and innerAngle(polygon.points[work[work[i].next].next],polygon.points[work[i].next],polygon.points[work[i].prev]):
            task.put(work[i].next)
            reflex.remove(work[i].next)
        work[work[i].prev].next=work[i].next
        work[work[i].next].prev=work[i].prev

    return triangles






        



        




