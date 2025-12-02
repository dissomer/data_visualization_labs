import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.animation import PillowWriter

FPS = 30
DURATION_SEC = 5
FRAMES = FPS * DURATION_SEC

center = (0.0, 0.0)
width = 4.0
height = 2.0
start_angle = 0.0
angular_speed = 360.0 / DURATION_SEC

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Animation')

ax.grid(True, linestyle='--', alpha=0.4)

ellipse = mpatches.Ellipse(xy=center, width=width, height=height,
                           angle=start_angle, facecolor='orange',
                           edgecolor='black', lw=2, animated=True)


def init():
    ax.add_patch(ellipse)
    return (ellipse,)


def animate(frame):
    angle = start_angle + angular_speed * (frame / FPS)
    ellipse.angle = angle % 360.0
    return (ellipse,)


anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=FRAMES,
                               interval=1000.0 / FPS,
                               blit=True)

output_filename = 'ellipse_rotation.gif'
writer = PillowWriter(fps=FPS)
anim.save(output_filename, writer=writer)

print(f'{output_filename}')
plt.show()
