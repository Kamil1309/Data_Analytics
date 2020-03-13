import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(1)

### 1 ###
def plot_histograms( *, mu, sigma ):
    x = mu + sigma * np.random.randn(10000)

    #cumulative=False
    plt.subplot(3, 2, 1)
    plt.hist(x, 10, density=True)
    plt.title('cumulative=False')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(x, 50, density=True)
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.hist(x, 200, density=True)
    plt.xlabel('Value')
    plt.grid(True)
    #cumulative=True
    plt.subplot(3, 2, 2)
    plt.hist(x, 10, density=True, cumulative=True)
    plt.title('cumulative=True')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(x, 50, density=True, cumulative=True)
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.hist(x, 200, density=True, cumulative=True)
    plt.xlabel('Value')
    plt.grid(True)

    plt.suptitle('Normal distribution, mu = %d, sigma = %d' %(mu, sigma))
    plt.show()

plot_histograms( mu= 50, sigma= 15)
plot_histograms( mu= 100, sigma= 15)
plot_histograms( mu= 100, sigma= 50)
