
import matplotlib.pyplot as plt
import Classes as cl

for j in range(20):
    temp=cl.Obstacle(20,20)
    temp.offset(100,100)
    for i in range(1,len(temp.points)):
        plt.plot((temp.points[i].x,temp.points[i-1].x),(temp.points[i].y,temp.points[i-1].y))

    plt.plot((temp.points[0].x,temp.points[len(temp.points)-1].x),(temp.points[0].y,temp.points[len(temp.points)-1].y))



plt.show()



def obstacleGen(w, h):
    obstacleGen(0,0)


def gen(w, h):
    obstacleGen(0,0)