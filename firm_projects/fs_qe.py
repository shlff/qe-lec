## Imports

import pandas as pd
import numpy as np
from numba import jit, prange, njit, jitclass, float64
import seaborn as sns
import matplotlib.pyplot as plt
import quantecon as qe
import statsmodels.api as sma
from scipy import stats
from scipy.stats import norm
import scipy.optimize as optimize
from statsmodels.stats.anova import anova_lm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.base.model import GenericLikelihoodModel
from scipy.integrate import quad, fixed_quad, simps


# data process

def import_data(file='data/jp1.csv'):
    df = pd.read_csv(file)
    df = df[['id', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']].set_index('id')
    return df

def import_data2(file='data_all/us_noe_all1.csv'):
    df = pd.read_csv(file)
    df = df[['id', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']].set_index('id')
    return df

def importdata(file='data/fortune500-2019.csv', year='2019'):
    df = pd.read_csv(file)
    df = df[['company', 'revenue ($ millions)']] * 10 ** 3
    df.rename(columns={'revenue ($ millions)': year}, inplace=True)
    return df
    
def importdata2(file='data/fortune1000-employee-2020.csv', year='2020'):
    df = pd.read_csv(file)
    df = df[['name', 'employee']] 
    df.rename(columns={'employee': year}, inplace=True)
    return df

def pre_process(file='data/usa.csv'):
    df = pd.read_csv(file)
    df1 = df[['id', 'year', 'tasset', 'fgrow_log']]
    df2 = df1[df1['fgrow_log'] != 0.0]
    df3 = df2.sort_values('tasset').reset_index(drop=True)
    df3.index += 1
    df3.index.name = 'count'
    df4 = df3.reset_index()
    return df4

def preprocessdata(df1):
    
    df2 = df1.stack()
    df3 = df2.to_frame().reset_index()
    df3.columns = 'id', 'year', 'tasset'
    df4 = df3.set_index(['id', 'year'])
    df4.loc[df4['tasset'] == 'n.a.'] = np.nan
    df4.loc[df4['tasset'] == '0'] = np.nan
    df4['tasset'] = df4['tasset'].str.replace(',', '').astype(float)
    
    return df4


def preprocess_data0(df4):
    
    df5 = df4.reset_index('id')
    df5['logtasset'] = np.log(df5['tasset'])
    df6 = df5[df5['logtasset'] != -np.inf]
    df6['fgrow_log'] = df6.groupby('id')['logtasset'].diff(periods=-1) * 100
    df7 = df6.dropna()
    
    return df7

def process45line(df, ID='id'):
    
    df1 = df.reset_index(ID)
    df1['logtasset'] = np.log(df1['tasset'])
    df1['tasset_prev'] = df1.groupby(ID)['tasset'].shift(+1)
    df1['logtasset_prev'] = df1.groupby(ID)['logtasset'].shift(+1)
    df2 = df1.dropna()
    
    return df2

def processdata(file='data/USA18.csv', year1='2018', year2='2019'):
    
    df = pd.read_csv(file)
    df = df[['id', year1, year2]].set_index('id')
    df0 = df.dropna()
    df1 = df0.stack()
    df2 = df1.to_frame().reset_index()
    df2.columns = 'id', 'year', 'noe'  # noe = number of employees
    df3 = df2.set_index(['id', 'year'])
    df3[df3['noe'] == 'n.a.'] = np.nan
    df3[df3['noe'] == '0'] = np.nan
    df3['noe'] = df3['noe'].astype(float)
    df4 = df3.reset_index('id')
    df4['lognoe'] = np.log(df4['noe'])
    df4['fgrow_log'] = -1 * df4.groupby('id')['lognoe'].diff(periods= -1) 
    df5 = df4.dropna().reset_index()
    
    return df5

def generate_equalbinsize(file='data_all/us_to_all_postprocess.csv', 
                          varold='logturnover', 
                          varnew='logturnover_mean',
                          equalbingap=True,  # equalbingap=True if we want bins with equal center gaps
                          n=25):
    df = pd.read_csv(file)
    df1 = df.sort_values(varold).reset_index(drop=True)
    df1.index += 1
    df1.index.name = 'count'
    df2 = df1.reset_index()
    if equalbingap is True:
        df2['quantile_ex'] = pd.cut(df2[varold], bins=n)
    else:
        df2['quantile_ex'] = pd.qcut(df2[varold], q=n)
            
    df3 = df2.groupby('quantile_ex').mean().reset_index()
    df4 = df2.groupby('quantile_ex').std().reset_index()
    df4[varnew] = df3[varold]
    return df4

def dataprocess(file='data/USA19-13_top1.csv', var1='lognoe', var2='noe', var3='noe_mean'):
    df = pd.read_csv(file)
    df1 = df.sort_values(var1).reset_index(drop=True)
    df1.index += 1
    df1.index.name = 'count'
    df2 = df1.reset_index()
    df2['quantile_ex_1'] = pd.qcut(df2[var1], q=4)
    df3 = df2.groupby('quantile_ex_1').mean().reset_index()
    df4 = df2.groupby('quantile_ex_1').std().reset_index()
    df4[var3] = df3[var2]
    
    return df, df2, df3, df4

def preprocess_data(df_us, var1='noe', var2='lognoe'):
    df_us1 = df_us.stack()
    df_us2 = df_us1.to_frame().reset_index()
    df_us2.columns = 'id', 'year', var1  # noe = number of employment
    df_us3 = df_us2.set_index(['id', 'year'])
    df_us3[df_us3[var1] == 'n.a.'] = np.nan
    df_us3[df_us3[var1] == '0'] = np.nan
    df_us3[var1] = df_us3[var1].astype(float)
    df_us4 = df_us3.reset_index('id')
    df_us4[var2] = np.log(df_us4[var1])
    df_us4['fgrow_log'] = -1 * df_us4.groupby('id')[var2].diff(periods=-1) 
    df_us5 = df_us4.dropna().reset_index()
    return df_us5



# estimations

def gabaix_est(s_dist, c=0.1, s=1/2):
    w = - np.sort(- s_dist)             # Reverse sort
    w = w[:int(len(w) * c)]             # extract top c * 100%
    rank_data = np.arange(len(w)) + 1 - s
    size_data = w
    y = np.log(np.array(rank_data))     # y = ln(i-s)
    x = np.log(np.array(size_data))     # x = ln(S_{(i)})
    x_addconstant = sma.add_constant(x)
    model = sma.OLS(y, x_addconstant)
    results = model.fit()
    
#     m, b = np.polyfit(y, x, 1)          # m = slope, b=intercept
#     print(m, b)

    b, m = results.params
#     print(b, m)
#     print(results.summary())
#     ax.plot(x, (m)*x + b)
#     ax.loglog(rank_data, size_data, '-o', markersize=3.0, alpha=0.5)
#     print("tail index is", -m)
    return -m

def hill_est(s_dist, c=0.1):
    w = - np.sort(- s_dist)                  # Reverse sort
#     print(w)
    w = w[:int(len(w) * c)]                # extract top c * 100%
    # rank_data = np.array(np.arange(len(w)) + 1)
    size_data = np.array(w)
    n = len(size_data)
#     print(rank_data)
#     print(size_data)
#     print(n)
    S = 0
    for i in range(n-1):
        S += np.log(size_data[i+1]) -  np.log(size_data[n-1])
    H = (n - 2) / S
    return H

def ranksizeplot_data(file='data/usa.csv'):

    df1 = pd.read_csv(file)
    s_dist1 = df1.tasset

    c = 1

    fig, ax = plt.subplots()

    # create rank-size data
    w = - np.sort(- s_dist1)                  # Reverse sort
    w = w[:int(len(w) * c)]                # extract top (c * 100)%
    rank_data = np.arange(len(w)) + 1
    size_data = w

    ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
    ax.set_xlabel("log rank")
    ax.set_ylabel("log size")
    ax.set_xlim(5000.0, 70000)
    ax.set_ylim(0, 1000000000000000000)
    plt.show()


def nega_log_like(paras, y, x):
    if (paras[2] + paras[3]) < 0:
        return 1e8
    
    like = norm.pdf(y, 
                    loc = paras[0] + paras[1] * x, 
                    scale = paras[2] + paras[3] * ((x) ** 2))
    
    if all(v == 0 for v in like):
        return 1e8
    
    return -sum(np.log(like[np.nonzero(like)]))

def nega_log_like_mu_b_fix(paras, y, x):
    if (paras[2] + paras[3]) < 0:
        return 1e8
    
    like = norm.pdf(y, 
                    loc = paras[0], 
                    scale = np.sqrt(paras[2] + paras[3] * ((x) ** 2)))
    
    if all(v == 0 for v in like):
        return 1e8
    
    return -sum(np.log(like[np.nonzero(like)]))

def alpha(μ, σ2):
    return - 2 * μ / σ2

def mle_mean_var_se(df, x='logtasset_inv', y='fgrow_log', x_zero=[0, 0, 10, 10]):

    X = df[x]
    Y = df[y]

    opt_res = optimize.minimize(fun = nega_log_like, 
                                x0 = x_zero, 
                                args = (Y, X))
    μ_σ = opt_res.x
    se = np.sqrt(np.diag(opt_res.hess_inv))
    
    return opt_res, μ_σ, se

@njit
def growth(s, paras):
    a = paras[0] + np.sqrt(paras[2]) * np.random.randn()
    b = paras[1] + np.sqrt(paras[3]) * np.random.randn()
    return a + b / np.log(s)

@njit
def growth2(s, paras, c=10):
    a = paras[0] + paras[2] * np.random.randn()
    b = paras[1] + paras[3] * np.random.randn()
    return a + b * max(0, c - np.log(s))

def gen_growth_obs(paras, num_obs=100_000, min_firm_size=10):
    size_obs = min_firm_size + np.random.pareto(1.0, size=num_obs)
    growth_obs = np.empty(num_obs)
    for i in range(num_obs):
        growth_obs[i] = growth(size_obs[i], paras)
        
    return size_obs, growth_obs

def sim_growth_plot(paras, M):
    s, g = gen_growth_obs(paras, num_obs=M)

    fig, ax = plt.subplots()
    ax.scatter(np.log(s), g)
    plt.show()


firm_dynamics_data = [
    ('μ_a',  float64),         # μ_a
    ('μ_b',    float64),       # μ_b
    ('σ2_a',    float64),   # σ2_a 
    ('σ2_b',    float64),   # σ2_b
    ('s_mean',    float64)
]

@jitclass(firm_dynamics_data)
class FirmDynamics:

    def __init__(self,
                 paras=[-5.32, -3.96, 10.085, 2176.81]):

        self.μ_a, self.σ2_a = paras[0], paras[2]
        self.μ_b, self.σ2_b = paras[1], paras[3]

        # Record stationary moments
        self.s_mean = np.exp( - self.μ_a / self.μ_b )


    def parameters(self):
        """
        Collect and return parameters.
        """
        parameters = (self.μ_a, self.σ2_a,
                      self.μ_b, self.σ2_b)
        return parameters

    def update_states(self, s):
        """
        Update one period, given current wealth w and persistent
        state z.
        """

        # Simplify names
        params = self.parameters()
        μ_a, σ2_a, μ_b, σ2_b = params
        
        a = μ_a + np.sqrt(σ2_a) * np.random.randn()
        b = μ_b + np.sqrt(σ2_b) * np.random.randn()

        # Update firm size
        sp = s * np.exp( a + b / np.log(s) )
        return sp

@njit
def firm_size_series(fd, s_0, n):
    """
    Generate a single time series of length n for firm sizes given
    initial value s_0.

    The initial persistent state s_0 for each firm is drawn from
    the stationary distribution of the law of motion.

        * fd: an instance of FirmDynamics
        * s_0: scalar
        * n: int

    """
    s = np.empty(n)
    s[0] = s_0
    for t in range(n-1):
        s[t+1] = fd.update_states(s[t])
    return s

@njit(parallel=True)
def update_cross_section(fd, s_distribution, shift_length=500):
    """
    Shifts a cross-section of firm forward in time

    * fd: an instance of FirmDynamics
    * s_distribution: array_like, represents current cross-section

    Takes a current distribution of wealth values as w_distribution
    and updates each s_t in s_distribution to s_{t+j}, where
    j = shift_length.

    Returns the new distribution.

    """
    new_distribution = np.empty_like(s_distribution)

    # Update each household
    for i in prange(len(s_distribution)):
        s = s_distribution[i]
        for t in range(shift_length-1):
            s = fd.update_states(s)
        new_distribution[i] = s
    return new_distribution

def sim_fss(fd, ts_length=10):
    s = firm_size_series(fd, fd.s_mean, ts_length)

    fig, ax = plt.subplots()
    ax.plot(s)
    plt.show()



# @njit
def growthrate1(s, paras):
    """
    It takes 
    
        s:       float; a firm-size value
        paras:   μ_α, μ_β, σ^2_α, σ^2_β
        
    and returns
        
        a new firm-size value 
    
    with the update equation (1) above.
        
    """
    a = paras[0] + np.sqrt(paras[2]) * np.random.randn()
    b = paras[1] + np.sqrt(paras[3]) * np.random.randn()
    return a + b /  (s ** (1 / 4))




# @njit
def growthrate2(s, paras):
    """
    It takes 
    
        s:       float; a firm-size value
        paras:   μ_α, μ_β, σ^2_α, σ^2_β
        
    and returns
        
        a new firm-size value 
    
    with the update equation (1) above.
        
    """
    a = paras[0] + np.sqrt(paras[2]) * np.random.randn()
    b = paras[1] + np.sqrt(paras[3]) * np.random.randn()
    return a + b / np.log(s)

# @njit
def growthrate3(s, paras):
    """
    It takes 
    
        s:       float; a firm-size value
        paras:   μ_α, μ_β, σ^2_α, σ^2_β
        
    and returns
        
        a new firm-size value 
    
    with the update equation (1) above.
        
    """
    a = paras[0] + np.sqrt(paras[2]) * np.random.randn()
    b = paras[1] + np.sqrt(paras[3]) * np.random.randn()
    return a + b * np.max(20 - np.log(s), 0)

def random_pareto(alpha, size=10, x_m=np.exp(8)):
    X = np.empty(size)
    for i in range(size):
        u = np.random.uniform()
        X[i] = x_m / (u ** (1/ alpha))
    return X


def gen_growth_obs_new(paras, df, model='1'):
    """
    It takes
    
       paras:           μ_α, μ_β, σ^2_α, σ^2_β
       num_obs:         the number of firms
       min_firm_size:   lower bound of the firm size
       
    and returns
    
       size_obs:        a series of discrete values for the firm size
       growth_obs:      a series of discrete values for the firm growth rate
       
       
    """
    
    size_obs = df['tasset'].array
    growth_obs = np.empty(len(size_obs))
    for i in range(len(size_obs)):
        if model == '1':
            growth_obs[i] = growthrate1(size_obs[i], paras) * 100
        if model == '2':
            growth_obs[i] = growthrate2(size_obs[i], paras) * 100
        if model == '3':
            growth_obs[i] = growthrate3(size_obs[i], paras) * 100
            
        
    return size_obs, growth_obs

def mean_growth1(s, paras):
    return paras[0] + paras[1] / (s ** (1/4))

def mean_growth2(s, paras):
    return paras[0] + paras[1] / (np.log(s))

def mean_growth3(s, paras):
    return paras[0] + paras[1] * (20 - np.log(s))

def sd_growth1(s, paras):
    return np.sqrt( paras[2] + paras[3] / (s ** (1/2)) )

def sd_growth2(s, paras):
    return np.sqrt( paras[2] + paras[3] / ((np.log(s)) ** 2) )

def sd_growth3(s, paras):
    return np.sqrt( paras[2] + paras[3] * ((20 - (np.log(s))) ** 2) )


def generating_pdf(draws, n=100):
    s = pd.Series(draws)
    x = s.value_counts()
    x1 = x.to_frame().reset_index()
    x1.columns = 'fgrow_log', 'freq'
    x1['quantile_ex_2'] = pd.cut(x1['fgrow_log'], n)
    x2 = x1.groupby('quantile_ex_2').sum().reset_index()
    x2['fgrow_mean'] = [n.mid for n in x2['quantile_ex_2']]
    x2['prob'] = x2['freq'] / x2['freq'].sum()
    return x2


# plots

def scatterplot(df, 
                x='tasset', 
                y='fgrow_log', 
                xlabel='Size (Total Assets)', 
                ylabel='Growth Rates (log)'):
    
    fig, ax = plt.subplots()
    df.plot.scatter(x, y, ax=ax)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize = 12)
    # ax.set_xlim(-100.0, 300000000)
    # ax.set_ylim(-2000.0, 2000.0)
    plt.show()



def plots(df, x='tasset_prev', y='tasset', reg=False):
    
    fig, ax = plt.subplots(figsize=(10,10))

    sns.regplot(x, y, data=df, 
                fit_reg=reg, scatter=True, ax=ax, 
                line_kws={'color':'cyan', 'label':'Regression Line'}, scatter_kws={'color':'blue'})

    ax.set_xlabel('Size (Total Assets) $S_t$', fontsize=12)
    ax.set_ylabel('Size (Total Assets) $S_{t+1}$', fontsize = 12)
    
    # 45 Degree Line and Fitting Polynomial
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, label='45 Degree Line')

    # Polynomial Approximation with Degree 3

    if reg == False:
        p1 = np.poly1d(np.polyfit(df.tasset_prev, df.tasset, 3))
        ax.plot(x, p1(x), label='Polynomial Approximation')


    lineStart = x.min() 
    lineEnd = x.max()  

    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.legend()
    plt.show()


def plot_fs_cross_sectional(fd1):
    
    num_firms = 200_000
    T = 200  # how far to shift forward in time
    s_0 = np.ones(num_firms) * fd1.s_mean

    s_star1 = update_cross_section(fd1, s_0, shift_length=T)

    # remove inf in s_star
    s_doublestar1 = s_star1[s_star1 < 1E308]
#     print(s_0, s_star1, s_doublestar1)

    # plot

    c = 1

    fig, ax = plt.subplots()

    # create rank-size data
    w = - np.sort(- s_doublestar1)                  # Reverse sort
    w = w[:int(len(w) * c)]                # extract top (c * 100)%
    rank_data = np.arange(len(w)) + 1
    size_data = w

    ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
    ax.set_xlabel("log rank")
    ax.set_ylabel("log size")
    # ax.set_xlim(5000.0, 70000)
    # ax.set_ylim(0, 1000000000000000000)
    plt.show()

def barplot(df, bar=2):
    df['quantile_ex'] = pd.qcut(df['tasset'], q=bar)
    df1 = df.groupby('quantile_ex').mean().reset_index()
    ax = df1.plot.bar('quantile_ex', 'fgrow_log')
    ax.set_xlabel('firm size bins',fontsize=14)
    ax.set_ylabel('growth mean $\%$',fontsize=14)

def simplot(opt_res, df, m='1'):
    
    # M = 200_000 # set the number of firms
    s, g = gen_growth_obs_new(opt_res.x, df=df, model=m)

    fig, ax = plt.subplots()
    ax.scatter(np.log(s), g)
    ax.set_xlabel('Log Size (Log Total Assets)', fontsize=12)
    ax.set_ylabel('Growth Rates (log)', fontsize = 12)
    ax.set_xlim(7.0, 23)
    plt.show()

def barplots2(df):
    ax = df.plot.bar('quantile_ex_1', 'fgrow_log')
    ax.set_xlabel('log firm size bins',fontsize=14)
    ax.set_ylabel('growth mean $\%$',fontsize=14)

def bslplot(df, y='fgrow_mean_sim1'):
    ax = df[['quantile_ex_1', y]].plot(x='quantile_ex_1', linestyle='-', marker='o', color='r')
    df[['quantile_ex_1', 'fgrow_log']].plot(x='quantile_ex_1', kind='bar', color='g', ax=ax)
    plt.show()

def plotbars(df, xlabel='log firm size bins', ylabel='growth mean'):
    ax = df.plot.bar('quantile_ex_1', 'fgrow_log')
    ax.set_xlabel(xlabel,fontsize=14)
    ax.set_ylabel(ylabel,fontsize=14)


def density_plot(x1, x2, x3, x4, y1, y2, y3, y4):
    
    fig, ax = plt.subplots()

    plt.plot(x1, y1, '-ok', color='r', label='bin1')
    plt.plot(x2, y2, '-ok', marker='s', color='lightgreen', label='bin2')
    plt.plot(x3, y3, '-ok', color='blue', markerfacecolor=None, label='bin3')
    plt.plot(x4, y4, '-ok', marker='^', color='purple', label='bin4')

    # ax.set_ylim(1e-3, 1e-1)
    ax.set_xlim(-0.75, 0.75)

    plt.xlabel('Growth of (absolute) firm-size', fontsize=14)
    plt.ylabel('Probability density', fontsize=14)

    ax.legend()
    plt.show()

def plot_bars(df, X='logturnover_mean', Y='fgrow_log', coefwidth=3):
    x = np.array(df[X])
    y = np.array(df[Y])
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    width = np.min(np.diff(x)) * coefwidth

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x,y,width, color='tab:blue', label='-Ymin')

    plt.xlabel('log firm size bins',fontsize=14)
    plt.ylabel('growth standard deviation',fontsize=14)
    plt.xticks(np.arange(2, 17, step=2))

    plt.show()

def plotlinedot(df, X='logturnover_mean', Y='fgrow_log'):
    
    x = np.array(df[X])
    y = np.array(df[Y])
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    fig, ax = plt.subplots()

    ax.plot(x, y, marker = 'o')

    plt.xlabel('log firm size bins',fontsize=14)
    plt.ylabel('growth standard deviation',fontsize=14)
    plt.xticks(np.arange(0, 21, step=2))

    plt.show()
































