# Fix LaTeX syntax error: should be \frac{1}{2}, not \frac12
import numpy as np
import matplotlib.pyplot as plt
# f_min range
f_min = np.linspace(0, 5, 500)

# Lower and upper bounds
lower_bound = 1 - np.exp(-f_min / 8)
upper_bound = np.sqrt((1/4) * f_min)

# Plot
plt.figure(figsize=(6,4))
plt.plot(f_min, lower_bound, label=r'Lower bound: $1 - e^{-f_{\min}/8}$', linewidth=2)
plt.plot(f_min, upper_bound, label=r'Upper bound: $\frac{1}{2}\sqrt{f_{\min}}$', linewidth=2)
plt.fill_between(f_min, lower_bound, upper_bound, color='gray', alpha=0.2, label='Possible TV range')

plt.xlabel(r'$f_{\min}$', fontsize=12)
plt.ylabel(r'Total Variation Distance', fontsize=12)
plt.title('TV Bounds as a Function of $f_{\\min}$', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)
plt.ylim(0, 1.1)
plt.savefig('tv_bounds_plot.pdf', bbox_inches='tight', dpi=300)
# plt.show()
