import numpy as np
import matplotlib.pyplot as plt


def scalar():
    x = np.linspace(0, 8, 200)
    y = np.linspace(0, 8, 200)
    X, Y = np.meshgrid(x, y)

    U = X * Y ** 2 - np.sqrt(X ** 3 * Y)

    dUx, dUy = np.gradient(U, x, y)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    pcm = ax[0].pcolormesh(X, Y, U, cmap='viridis', shading='auto')
    ax[0].set_title(r'1. Scalar field $u(x,y)=xy^2-\sqrt{x^3y}$')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    fig.colorbar(pcm, ax=ax[0])
    skip = (slice(None, None, 8), slice(None, None, 8))
    ax[1].quiver(X[skip], Y[skip], dUx[skip], dUy[skip], color='black')
    ax[1].set_title('Gradient field ∇u(x,y)')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()


def vector_2d():
    x = np.linspace(-8, 8, 25)
    y = np.linspace(-8, 8, 25)
    X, Y = np.meshgrid(x, y)
    U = X ** 3
    V = -Y ** 3

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.quiver(X, Y, U, V, color='darkred')
    plt.title('2. 2D vector field F=(x³, −y³)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.streamplot(X, Y, U, V, color=np.hypot(U, V), cmap='plasma')
    plt.title('Streamlines of F(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def vector_3d():
    x = np.linspace(-7, 7, 10)
    y = np.linspace(-7, 7, 10)
    z = np.linspace(-7, 7, 10)
    X, Y, Z = np.meshgrid(x, y, z)

    denom = X ** 3 + Y ** 3 + Z ** 3
    denom[denom == 0] = np.nan

    U = X ** 2 / denom
    V = Y ** 2 / denom
    W = Z ** 2 / denom

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.4, normalize=True, color='teal')
    ax.set_title('3. 3D vector field F(x,y,z)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.show()


def tensor():
    x = np.linspace(-2, 2, 5)
    y = np.linspace(-2, 2, 5)
    z = np.linspace(-2, 2, 5)
    X, Y, Z = np.meshgrid(x, y, z)

    eps = 1e-6
    T = np.zeros((3, 3, *X.shape))
    T[0, 0] = np.sin(X + Y) / (X**2 + Y**2 + Z**2 + eps)
    T[0, 1] = X
    T[0, 2] = Y
    T[1, 1] = np.cos(Y + Z) / (X**3 + Y**3 + Z**3 + eps)
    T[1, 2] = Z
    T[2, 2] = np.cos(X + Z) / (X**3 + Y**3 + Z**3 + eps)

    glyph_types = {
        0: 'Cuboid',
        1: 'Cylinder',
        2: 'Ellipsoid',
        3: 'Kindlmann Superquadric'
    }

    fig = plt.figure(figsize=(16, 10))
    axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]

    def cuboid(ax, center, size, color):
        cx, cy, cz = center
        sx, sy, sz = size
        r = np.array([[cx-sx, cx+sx],
                      [cy-sy, cy+sy],
                      [cz-sz, cz+sz]])
        for s, e in [
            ([0, 0, 0], [1, 0, 0]), ([0, 1, 0], [1, 1, 0]),
            ([0, 0, 1], [1, 0, 1]), ([0, 1, 1], [1, 1, 1]),
            ([0, 0, 0], [0, 1, 0]), ([1, 0, 0], [1, 1, 0]),
            ([0, 0, 1], [0, 1, 1]), ([1, 0, 1], [1, 1, 1]),
            ([0, 0, 0], [0, 0, 1]), ([1, 0, 0], [1, 0, 1]),
            ([0, 1, 0], [0, 1, 1]), ([1, 1, 0], [1, 1, 1])
        ]:
            s = [r[0][s[0]], r[1][s[1]], r[2][s[2]]]
            e = [r[0][e[0]], r[1][e[1]], r[2][e[2]]]
            ax.plot3D(*zip(s, e), color=color, linewidth=0.8)

    def random_rotation_matrix():
        a, b, g = np.random.uniform(0, 2 * np.pi, 3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(a), -np.sin(a)],
                       [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)],
                       [0, 1, 0],
                       [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(g), -np.sin(g), 0],
                       [np.sin(g), np.cos(g), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def cylinder(ax, center, radius, height, color):
        cx, cy, cz = center
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(-height/2, height/2, 10)
        theta, z = np.meshgrid(theta, z)
        X = radius * np.cos(theta)
        Y = radius * np.sin(theta)
        Z = z

        R = random_rotation_matrix()
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
        Xr, Yr, Zr = R @ pts
        Xr, Yr, Zr = Xr.reshape(X.shape), Yr.reshape(Y.shape), Zr.reshape(Z.shape)

        ax.plot_surface(Xr + cx, Yr + cy, Zr + cz, color=color, alpha=0.6, linewidth=0)

    def ellipsoid(ax, center, radii, color):
        cx, cy, cz = center
        rx, ry, rz = radii
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        U, V = np.meshgrid(u, v)
        X = cx + rx * np.cos(U) * np.sin(V)
        Y = cy + ry * np.sin(U) * np.sin(V)
        Z = cz + rz * np.cos(V)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.6, linewidth=0)

    def kindlmann_superquadric(ax, center, tensor, color=(0.4, 0.6, 1.0, 0.7)):
        t = 0.5 * (tensor + tensor.T)
        w, v = np.linalg.eigh(t)
        w = np.abs(w) + 1e-6
        w /= np.max(w)
        lam1, lam2, lam3 = np.sort(w)[::-1]

        R_L = (lam1 - lam2) / (lam1 + lam2 + lam3)
        R_P = (lam2 - lam3) / (lam1 + lam2 + lam3)
        eps1 = 2 / (1 + R_L)
        eps2 = 2 / (1 + R_P)

        u = np.linspace(-np.pi/2, np.pi/2, 15)
        v_ang = np.linspace(-np.pi, np.pi, 15)
        U, Vv = np.meshgrid(u, v_ang)
        sgn = np.sign
        powabs = lambda x, p: sgn(x) * (np.abs(x)**p)

        X = powabs(np.cos(U), 2/eps1) * powabs(np.cos(Vv), 2/eps2)
        Y = powabs(np.cos(U), 2/eps1) * powabs(np.sin(Vv), 2/eps2)
        Z = powabs(np.sin(U), 2/eps1)

        scale = 0.4
        S = np.diag([lam1, lam2, lam3]) * scale
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
        pts = v @ S @ pts
        Xp, Yp, Zp = pts.reshape(3, *X.shape)

        cx, cy, cz = center
        ax.plot_surface(Xp + cx, Yp + cy, Zp + cz, color=color, alpha=0.85, linewidth=0)

    glyph_radius = 0.25

    for gid, ax in enumerate(axes):
        name = glyph_types[gid]
        ax.set_title(name, fontsize=12)

        for i in range(x.size):
            for j in range(y.size):
                for k in range(z.size):
                    center = [x[i], y[j], z[k]]
                    tensor = T[:, :, i, j, k]
                    color = (0.4, 0.6, 1.0, 0.7)

                    if gid == 0:
                        cuboid(ax, center, (glyph_radius,)*3, color)
                    elif gid == 1:
                        cylinder(ax, center, glyph_radius, 0.6, color)
                    elif gid == 2:
                        ellipsoid(ax, center, (glyph_radius,)*3, color)
                    elif gid == 3:
                        kindlmann_superquadric(ax, center, tensor, color)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(20, 30)

    fig.suptitle('4. Tensor field: cuboid, cylinders, ellipsoid, Kindlmann superquadric', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    scalar()
    vector_2d()
    vector_3d()
    tensor()


if __name__ == "__main__":
    main()
