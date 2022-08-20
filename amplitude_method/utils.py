import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sympy import latex
from IPython.display import display, Latex


def hprint(header, obj):
    display(Latex(r"$" + header + latex(obj)+r"$") )
    
def lprint(latex_str, *elements):
    s = latex_str
    for i, e in enumerate(elements):        
        s = s.replace(f"{{{i}}}", r"$" + e +r"$")
    display(Latex(s))
    
    
def plot_quadratic(b, c, ax, title):
    
    def eq(x):
        return x**2 - b * x + c
    # Calulates parameters
    λ_m = b/2
    Δ = λ_m**2 - c
    if Δ >= 0:
        λ_n = λ_m - np.sqrt(Δ)
        λ_p = λ_m + np.sqrt(Δ)
    else:
        λ_n = λ_m - np.sqrt(-Δ)
        λ_p = λ_m + np.sqrt(-Δ)
    λs = np.linspace(λ_n - .2, λ_p + .2, num=100)
    lims = [np.min(λs), np.max(λs) ]
    lims_y = [np.min(eq(λs)), np.max(eq(λs)) ]
    length = lims[1] - lims[0]
    length_y = lims_y[1] - lims_y[0]
    f_p_max = eq(lims[1])
    f_m = eq(λ_m)
    # Plots
    ax.plot(λs, eq(λs))
    
    ax.axis('off')
    # X axis
    ax.arrow(min(lims[0], -0.1), 0, 
              max(max(length, lims[1]),0.1-min(lims[0], -0.1)), 0, 
              width = 0.01, 
              head_width = 0.05,
              alpha=.8)
    # X axis label
    ax.annotate(r"$\lambda$", 
                xy=(max(lims[1], 0.2), 0.1),
                textcoords='data',
               fontsize=18)
    # Y axis
    ax.arrow(0, min(lims_y[0], -0.2), 
              0, max(length_y, lims_y[1]), 
              width = 0.01,
              head_width = 0.05,
              alpha=.8)
    # Y axis label
    ax.annotate(r"f($\lambda$)", 
                xy=( 0.05, max(f_m, lims_y[1])),
                textcoords='data',
               fontsize=18)
    # f(λ_m)
    ax.vlines(λ_m, 0, f_m,'r', '--', alpha=.4)    
    
    ax.annotate(r"$\lambda_m = \frac{b}{2}$", xy=(λ_m, 0.0),
             xycoords='data',
             xytext=(λ_m*1.3, f_m*0.5),
             textcoords='data',
             fontsize=14,
             arrowprops=dict(arrowstyle= '-|>',
                             color='blue',
                             lw=2.5,
                             ls='--')
           )
    ax.annotate(r"$f(\lambda_m)$", xy=(λ_m*0.95, f_m*0.5), 
            xytext=(λ_m*.7, f_m*0.5), 
            xycoords='data', 
            textcoords='data',
            fontsize=14,
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle=f'->, widthB=2.0, lengthB=2.0, angleB=0', lw=1.0)
        )
    
    if Δ >= 0:
        ax.annotate(r"$\lambda_{-} = \lambda_m - \sqrt{\lambda_m^2 - c}$", xy=(λ_n, 0.0),
             xycoords='data',
             xytext=(λ_m - length/2.5, f_p_max/2),
             textcoords='data',
             fontsize=14,
             arrowprops=dict(arrowstyle= '-|>',
                             color='blue',
                             lw=2.5,
                             ls='--')
           )

        ax.annotate(r"$\lambda_{+} = \lambda_m + \sqrt{\lambda_m^2 - c}$", xy=(λ_p, 0.0),
             xycoords='data',
             xytext=(λ_m + length/14, f_p_max/2),
             textcoords='data',
             fontsize=14,
             arrowprops=dict(arrowstyle= '-|>',
                             color='blue',
                             lw=2.5,
                             ls='--')
           )
    #Title
    ax.text(λ_m - length/12, lims_y[1] *1.05, title, color='black', fontsize=14,
        bbox=dict(boxstyle='square,pad=.5',facecolor='none', edgecolor='red'))

def plot_two_levels(ax, 
                    domain, 
                    extent, 
                    xlabel, ylabel,
                    level,
                    level_names,
                    color='green'
                   ):
    d = domain.copy()
    d[d <= level] = -1
    d[d > level] = 0
    level_nums = 0
    if np.all(d == -1):
        cmap = colors.ListedColormap(['white'])
        level_nums = 0
    elif np.all(d == 0):
        cmap = colors.ListedColormap([color])
        level_nums = 1
    else:
        cmap = colors.ListedColormap(['white', color])
        level_nums = 2
    img = ax.imshow(d,
                    extent=extent,
                    origin='lower',
                    cmap=cmap,
                    alpha=.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(img, ax=ax)
    if level_nums == 0:
        cbar.set_ticks([-1])
        cbar.set_ticklabels([level_names[0]])
    elif level_nums == 1:
        cbar.set_ticks([0])
        cbar.set_ticklabels([level_names[1]])
    else:
        cbar.set_ticks([-1, 0])
        cbar.set_ticklabels(level_names)
    plt.grid()