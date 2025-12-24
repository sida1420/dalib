import math
import matplotlib.pyplot as plt
n=4
theta=2*math.pi/n

x, y = 0, 0
radius = 1

fig, ax = plt.subplots(figsize=(6, 6))

# 2. Create the circle patch
# (x, y) is the center, followed by the radius
circle = plt.Circle((x, y), radius, color='blue', fill=False, linewidth=2, label='Radius = 1')
ax.add_patch(circle)
for i in range(n):
    rea=math.cos(theta*i)
    ima=math.sin(theta*i)
    print(complex(round(rea,2),round(ima,2)))
    

    plt.plot(rea, ima, 'ro', markersize=10)

    plt.annotate(f'({round(rea,2)}, {round(ima,2)})', 
             xy=(rea, ima),           # The point to annotate
             xytext=(rea + 0.1, ima + 0.1), # Where to place the text
             fontsize=12,
             color='darkred')

ax.set_aspect('equal')

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Real-axis')
plt.ylabel('Imagine-axis')
plt.title('Displaying a Point on a Graph')

# 5. Show or save the plot
plt.show()