import numpy as np
import matplotlib.pyplot as plt


def function_plots():
    x_y = np.linspace(-2.5, 2.5, 1000)
    y = 7 * np.sin(np.pi * x_y) - np.cos(3 * np.pi * x_y) * np.sin(np.pi * x_y)

    x_z = np.linspace(-4.0, 4.0, 1200)
    z = np.where(
        x_z <= 0,
        np.sqrt(1.0 + np.abs(x_z)) / (2.0 + np.abs(x_z)),
        (1.0 + x_z) / (2.0 + np.cos(x_z) ** 3),
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    ax1.plot(x_y, y, label=r'y(x)')
    ax1.set_title('1. Function: y(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.legend()
    ax1.text(0.02, 0.93,
             'y = 7·sin(πx) − cos(3πx)·sin(πx)',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    ax2.plot(x_z, z, label='z(x)', color='tab:orange')
    ax2.set_title('Function: z(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.grid(True)
    ax2.legend()
    ax2.text(0.02, 0.93,
             "z(x) = sqrt(1 + |x|) / (2 + |x|),  x ≤ 0\n"
             "z(x) = (1 + x) / (2 + cos³(x)),     x > 0",
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()
    plt.show()


def surface_plot():
    x = np.linspace(0, 10, 200)
    y = np.linspace(-1.0, 1.0, 200)
    X, Y = np.meshgrid(x, y)
    Z = 5 * Y * (np.cos(X - 5) ** 2) - 5 * (Y ** 3) * np.exp(Y + 1)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_title(r'2. Surface $z=5y\cos^2(x-5)-5y^3e^{(y+1)}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(surf, shrink=0.6)
    ax.grid(True)
    plt.show()


def parametric_plot(a=1.0):
    t = np.linspace(-10, 10, 1000)
    x = t
    y = (a ** 3) / (t ** 2 + a ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=rf'$x=t,\ y=\dfrac{{a^3}}{{t^2+a^2}}\ (a={a})$')
    plt.title('3. Parametric')
    plt.xlabel('x (t)')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.text(0.02, 0.95, rf'$y=\dfrac{{{a:.0f}^3}}{{t^2+{a:.0f}^2}}$', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.08))
    plt.show()


def second_order_surface(a=2.0, b=3.0, c=1.0):
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    inside = (X ** 2) / (a ** 2) + (Y ** 2) / (b ** 2)
    Zpos = c * np.sqrt(inside)
    Zneg = -Zpos

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf1 = ax.plot_surface(X, Y, Zpos, cmap='coolwarm', alpha=0.9, linewidth=0, antialiased=True)
    ax.set_title(r'4. 2nd order surface $\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=0$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(surf1, shrink=0.6)
    ax.grid(True)
    plt.show()


def bar_chart():
    import numpy as np
    import matplotlib.pyplot as plt

    countries = ['USA', 'Germany', 'France', 'Italy', 'USSR']
    years = np.array([1900, 1913, 1929, 1938, 1950, 1960, 1970, 1980, 1990, 2000])

    data = {
        'USA':     [43, 56, 69, 76.5, 93.5, 105, 128.5, 146, 157.5, 175],
        'Germany': [16, 19, 20, 21.5, 23, 29, 37, 40.5, 46.5, 52.5],
        'France':  [21.5, 22, 22.5, 23, 23.5, 29.5, 47, 53, 65, 76.5],
        'Italy':   [13.5, 14.5, 16, 17, 18.5, 30.5, 42, 44.5, 49, 56],
        'USSR':    [37, 50.5, 58.8, 63, 75, 81.5, 87.5, 98, 120, 100]
    }

    plt.figure(figsize=(13, 7))
    bar_width = 0.10
    x = np.arange(len(years))

    for i, country in enumerate(countries):
        offset = (i - len(countries) / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, data[country], width=bar_width, label=country)

        for xi, yi in zip(x + offset, data[country]):
            plt.text(xi, yi + 2, f'{yi}', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.title('5. Global Agricultural Production (Added Value, Billion $)', fontsize=13)
    plt.xlabel('Year')
    plt.ylabel('Billion Dollars (2000 prices)')
    plt.xticks(x, years)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Country')
    plt.tight_layout()
    plt.show()


def main():
    function_plots()
    surface_plot()
    parametric_plot(a=1.0)
    second_order_surface(a=2.0, b=3.0, c=1.0)
    bar_chart()


if __name__ == "__main__":
    main()
