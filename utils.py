import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.stats as stats

# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols

plt.rcParams['figure.figsize'] = (8.0, 6.0)

# this is so the variable descriptions won't be truncated
pd.set_option('display.max_colwidth', None)

# load the cleaned data
df = pd.read_csv('cleaned_data/cleaned_data.csv')

# load the variable description key as a convenience
var_key = pd.read_csv('cleaned_data/variable_descriptions.csv')

# a map of state names to abbreviations
state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# a map of predictors for convenience
predictors = {
    'ses': [
        'gdp', 
        'pct_ge_bach', 
        'pct_poverty', 
        'med_hh_inc',
        'unemp_rate'
    ],
    'food_sec': [
        'conv_to_groc_ratio',
        'pct_low_store_access',
        'snap_per_capita'
    ],
    'all': [
        'gdp', 
        'pct_ge_bach', 
        'pct_poverty', 
        'med_hh_inc',
        'unemp_rate',
        'conv_to_groc_ratio',
        'pct_low_store_access',
        'snap_per_capita',
    ]
}

# the dependent variables
dependent_vars = [
    'pct_covid_vax',
    'pct_hisp_vax',
    'pct_native_vax',
    'pct_asian_vax',
    'pct_black_vax',
    'pct_isl_vax',
    'pct_white_vax'
]

def show_heatmap(dependent_var):
    '''
    Helper function to quickly visualize correlation between the 
    predictor variables and a single dependent variable.
    
    args:
        dependent_var (string): the dependent var to use
    
    side effects:
        displays a heatmap of the correlation between all of the
        predictor variables and the single dependent variable
    '''
    # get a correlation series in descending sorted order
    sorted_corr = df[[
        *predictors['all'], dependent_var
    ]].corr()[[dependent_var]].sort_values(by=[dependent_var], ascending=False)

    # mask to exclude the self-correlation for the dependent var
    mask = np.zeros_like(sorted_corr, dtype=int)
    mask[0] = 1

    hm = sns.heatmap(
        sorted_corr,

        # set the min and max so that we won't be misled by bright colors
        vmin=-1.0,
        vmax=1.0,

        # the mask from above
        mask=mask,

        # get the y-labels, exclude the dependent variable to avoid showing the case 
        # where correlation is 1.0 with itself
        yticklabels = pd.Series(['']).append(pd.Series(sorted_corr.index[1:]), ignore_index=True),

        annot=True);
    plt.show()

def show_scatter(predictor, demographic=['pct_white_vax', 'pct_black_vax', 'pct_hisp_vax', 'pct_asian_vax']):
    '''
    Generate a scatter plot of the Covid-19 vaccination rates for
    the four most numerous demographics, against a single predictor.
    
    args:
        predictor (string): the predictor variable to use
        demographic (list of strings): the list of demographics to include in the plot
        
    side effects:
        displays a scatterplot of the predictor variable against the 
        given demographics
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    
    for d in demographic:
        ax.scatter(
            df[predictor],
            df[d],  
            label=d,
            alpha=0.3
        )

    ax.set_xlabel(predictor)
    ax.set_ylabel('Vaccination rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def get_anova_table(dependent_var, predictors=predictors['all'], alpha=1.0):
    '''
    Get an ANOVA table using specified predictors and dependent variable.
    
    args:
        dependent_var (string): the dependent variable to use
        predictors (list of strings): the list of predictor variables to use
        alpha (float): the alpha parameter
        
    Returns:
        pandas DataFrame containing only rows where p is less than alpha
    '''
    ols_str = dependent_var + ' ~ ' + " + ".join(predictors)
        
    model = ols(ols_str, data=df).fit()
    
    table = sm.stats.anova_lm(model, typ=2)
    
    return table[table['PR(>F)'] < alpha]

def show_anova_table(dependent_var, predictors=predictors['all'], alpha=1.0):
    '''
    Display an ANOVA table using the specified predictors and dependent variable.
    
    args:
        dependent_var (string): the dependent variable to use
        predictors (list of strings): the list of predictor variables to use
        alpha (float): the alpha parameter
    
    side effects:
        Displays an ANOVA table with rows where p is less than alpha
    '''
    display(get_anova_table(dependent_var, predictors, alpha))
    
def get_most_useful_predictors(dependent_var, predictors=predictors['all'], alpha=0.01):
    '''
    Get a list of the most useful features for predicting the given 
    dependent variable.
    
    args:
        dependent_var (string): the dependent variable to use
        predictors (list of strings): the list of predictor variables to use
        alpha (float): the alpha parameter
        
    returns:
        list of strings: a list of the features that are deemed the most
        significant by ANOVA analysis
    '''
    return list(get_anova_table(dependent_var, predictors, alpha).index)

# a dict to store the most useful predictors as determined by ANOVA test
most_useful_predictors = dict([(dv, get_most_useful_predictors(dependent_var=dv)) for dv in dependent_vars])

def show_ols_summary(dependent_var):
    '''
    Generate an Ordinary Least Squares summary for the given dependent
    variable and the most useful predictive features for that variable.
    
    args:
        dependent_var (string): the dependent variable to use
        
    side effects:
        displays the OLS summary
    '''
    ols = sm.OLS(df[dependent_var], sm.add_constant(df[most_useful_predictors[dependent_var]]))
    ols_result = ols.fit()
    display(ols_result.summary().tables[0])
    display(ols_result.summary().tables[1])
    
def get_ols_R_sqr(dependent_var):
    '''
    Get the R^2 value of the ols regression model for the given dependent variable.
    '''
    ols = sm.OLS(df[dependent_var], sm.add_constant(df[most_useful_predictors[dependent_var]]))
    ols_result = ols.fit()
    return round(ols_result.rsquared, 3)
    
def get_ols_prob_f_stat(dependent_var):
    '''
    Get the Prob (F-statistic) value of the ols regression model for the given dependent variable.
    '''
    ols = sm.OLS(df[dependent_var], sm.add_constant(df[most_useful_predictors[dependent_var]]))
    ols_result = ols.fit()
    return f'{ols_result.f_pvalue:.1e}'

def show_final_summary_table():
    '''
    Print a summary table of the variables used, the regression coefficient, and the 
    probability of the F-statistic occuring by chance for each demographic.
    '''
    d = dict()

    # generate lists of 'Xs' where the predictor is useful
    for dv in dependent_vars:
        d[dv] = ['X' if p in most_useful_predictors[dv] else '' for p in predictors['all']]

    # transpose the data so it's the right way round
    fst = pd.DataFrame(d).T
    
    # name the columns using our list of predictors
    fst.columns = predictors['all']
    
    # add R^2 and f-test columns
    fst['R^2'] = [get_ols_R_sqr(dv) for dv in dependent_vars]
    fst['prob (F-stat)'] = [get_ols_prob_f_stat(dv) for dv in dependent_vars]
    
    fst.index.name = 'Demographic'
    
    # use longer and more descriptive names
    fst.rename(index={
        'pct_covid_vax': 'General population',
        'pct_hisp_vax': 'Hispanic',
        'pct_native_vax': 'American Indian / Alaska Native',
        'pct_asian_vax': 'Asian',
        'pct_black_vax': 'Black',
        'pct_isl_vax': 'Native Hawaiian / Pacific Islander',
        'pct_white_vax': 'White'
    }, 
    columns={
        'gdp': 'County GDP',
        'pct_ge_bach': 'Education level',
        'pct_poverty': 'Poverty rate',
        'med_hh_inc': 'Median household income',
        'unemp_rate': 'Unemployment rate',
        'conv_to_groc_ratio': 'Conv. store to grocery ratio',
        'pct_low_store_access': 'Low food store access',
        'snap_per_capita': 'SNAP (food stamps)'
    }, inplace=True)

    display(fst)