import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit, expit


# Fixing random state for reproducibility
np.random.seed(1)

### 1 ###
def plot_histograms( *, mu, sigma ):
    x = mu + sigma * np.random.randn(10000)
    print(x)
    #cumulative=False
    plt.subplot(3, 2, 1)
    plt.hist(x, 10, density=True)
    plt.title('cumulative=False')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(x, 40, density=True)
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.hist(x, 100, density=True)
    plt.xlabel('Value')
    plt.grid(True)
    #cumulative=True
    plt.subplot(3, 2, 2)
    plt.hist(x, 10, density=True, cumulative=True)
    plt.title('cumulative=True')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(x, 40, density=True, cumulative=True)
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.hist(x, 100, density=True, cumulative=True)
    plt.xlabel('Value')
    plt.grid(True)

    plt.suptitle('Normal distribution, mu = %d, sigma = %d' %(mu, sigma))
    plt.show()

# plot_histograms( mu= 0, sigma= 2)
# plot_histograms( mu= 1, sigma= 2)
# plot_histograms( mu= 1, sigma= 4)


### 2 ###
def plot_histograms_legit( *, mu, sigma ):
    x = mu + sigma * np.random.randn(10000)
    x = expit(x)
    print(x)
    #cumulative=False
    plt.subplot(3, 2, 1)
    plt.hist(x, 10, density=True)
    plt.title('cumulative=False')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(x, 40, density=True)
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.hist(x, 100, density=True)
    plt.xlabel('Value')
    plt.grid(True)
    #cumulative=True
    plt.subplot(3, 2, 2)
    plt.hist(x, 10, density=True, cumulative=True)
    plt.title('cumulative=True')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(x, 40, density=True, cumulative=True)
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.hist(x, 100, density=True, cumulative=True)
    plt.xlabel('Value')
    plt.grid(True)

    plt.suptitle('Normal distribution, mu = %d, sigma = %d' %(mu, sigma))
    plt.show()

# plot_histograms_legit( mu= 0, sigma= 2)
# plot_histograms_legit( mu= 1, sigma= 2)
# plot_histograms_legit( mu= 1, sigma= 4)

### 3 ###
def plot_histograms_poisson( mu ):
    x = np.random.poisson(mu, 10000)
    print(x)
    #cumulative=False
    plt.subplot(3, 2, 1)
    plt.hist(x, np.linspace(0, mu*3, mu), density=True)
    plt.title('cumulative=False')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(x, np.linspace(0, round(mu*2.5), round(mu*1.5)), density=True)
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.hist(x, np.linspace(0, round(mu*2.5), round(mu*2.5)), density=True)
    plt.xlabel('Value')
    plt.grid(True)
    #cumulative=True
    plt.subplot(3, 2, 2)
    plt.hist(x, np.linspace(0, mu*3, mu), density=True, cumulative=True)
    plt.title('cumulative=True')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(x, np.linspace(0, round(mu*2.5), round(mu*1.5)), density=True, cumulative=True)
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.hist(x, np.linspace(0, round(mu*2.5), round(mu*2.5)), density=True, cumulative=True)
    plt.xlabel('Value')
    plt.grid(True)

    plt.suptitle('Poisson distribution, mu = %d,' %mu)
    plt.show()

# plot_histograms_poisson( 5 )
# plot_histograms_poisson( 10 )
# plot_histograms_poisson( 15 )

### 4 ###
def plot_histograms_beta_normal( a, b ):
    x = np.random.beta(a, b, 10000)
    print(x)
    #cumulative=False
    plt.subplot(3, 2, 1)
    plt.hist(x, 10, density=True)
    plt.title('cumulative=False')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(x, 40, density=True)
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.hist(x, 100, density=True)
    plt.xlabel('Value')
    plt.grid(True)
    #cumulative=True
    plt.subplot(3, 2, 2)
    plt.hist(x, 10, density=True, cumulative=True)
    plt.title('cumulative=True')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(x, 40, density=True, cumulative=True)
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.hist(x, 100, density=True, cumulative=True)
    plt.xlabel('Value')
    plt.grid(True)

    plt.suptitle('Poisson distribution, natural parametrization')
    plt.show()

# plot_histograms_beta_normal( 2, 0.5 )
# plot_histograms_beta_normal( 0.5, 2 )
# plot_histograms_beta_normal( 0.5, 0.4 )
# plot_histograms_beta_normal( 2, 4 )

def plot_histograms_beta_location( l, d ):
    a = (1 - d)*l/d
    b = (1 - d)*(1-l)/d
    x = np.random.beta(a, b, 10000)
    print(x)
    #cumulative=False
    plt.subplot(3, 2, 1)
    plt.hist(x, 10, density=True)
    plt.title('cumulative=False')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(x, 40, density=True)
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.hist(x, 100, density=True)
    plt.xlabel('Value')
    plt.grid(True)
    #cumulative=True
    plt.subplot(3, 2, 2)
    plt.hist(x, 10, density=True, cumulative=True)
    plt.title('cumulative=True')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.hist(x, 40, density=True, cumulative=True)
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.hist(x, 100, density=True, cumulative=True)
    plt.xlabel('Value')
    plt.grid(True)

    plt.suptitle('Poisson distribution, natural parametrization')
    plt.show()

plot_histograms_beta_location( 0.1, 0.1 )
plot_histograms_beta_location( 0.1, 0.2  )
plot_histograms_beta_location( 0.5, 0.1  )
plot_histograms_beta_location( 0.5, 0.5  )