import numpy as np
import matplotlib.pyplot as plt
from sympy import latex
from IPython.display import display, Latex

def hprint(header, obj):
    display(Latex(r"$" + header + latex(obj)+r"$") )
    
def lprint(latex_str, *elements):
    s = latex_str
    for i, e in enumerate(elements):        
        s = s.replace(f"{{{i}}}", r"$" + e +r"$")
    display(Latex(s))
    
    
def plot_quadratic(b, c, ax):
    
    def eq(x):
        return x**2 - b * x + c
    
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
    length = lims[1] - lims[0]
    f_p_max = eq(lims[1])
    
    
    plt.plot(λs, eq(λs))
    
    plt.hlines(0, lims[0], lims[1], 'r', '--', alpha=.4)
    f_m = eq(λ_m)
    plt.vlines(λ_m, 0, f_m,'r', '--', alpha=.4)
    plt.xlim(lims)
    #plt.xlim([-2, 2])
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
             xytext=(λ_m - length/3, f_p_max/2),
             textcoords='data',
             fontsize=14,
             arrowprops=dict(arrowstyle= '-|>',
                             color='blue',
                             lw=2.5,
                             ls='--')
           )

        ax.annotate(r"$\lambda_{+} = \lambda_m + \sqrt{\lambda_m^2 - c}$", xy=(λ_p, 0.0),
             xycoords='data',
             xytext=(λ_m + length/10, f_p_max/2),
             textcoords='data',
             fontsize=14,
             arrowprops=dict(arrowstyle= '-|>',
                             color='blue',
                             lw=2.5,
                             ls='--')
           )
    
    plt.grid()
    plt.xlabel(r"$\lambda$", fontsize=14)
    plt.ylabel(r"$f(\lambda)$", fontsize=14)