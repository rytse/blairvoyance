import pandas as pd
import numpy as np
import scipy.cluster
import scipy.interpolate
# from scipy import stats
# from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
import matplotlib.pyplot as plt
from datetime import date
import thinkbayes2
import datapull

plt.style.use('ggplot')

WEIGHTS = {'A': .177,
           'B': .151,
           'C': .130,
           'D': .077}

DAYDIV = 167

class Electorate(thinkbayes2.Suite):
    '''Represents hypotheses about the state of the electorate'''
    
    def Likelihood(self, data, hypo):
        '''
        Likelihood of the data under the hypothesis.

        hypo: fraction of the population
        data: poll results
        '''
        bias, std, result = data
        error = result - hypo
        like = thinkbayes2.EvalNormalPdf(error, bias, std)
        return like


class PollAggregator:
    '''Estimator of the poll distributions'''
    
    
    def __init__(self, N=1000):
        self.suite = Electorate(np.linspace(0, 1, N))
        self.poll_vals = []
        self.poll_sizes = []
        self.data = set()
    
    
    def update(self, val, grade, nsamp, days):
        self.poll_vals.append(val)
        self.poll_sizes.append(nsamp)
        
        pvs = np.array(self.poll_vals)
        pss = np.array(self.poll_sizes)
        stdev_shift = np.mean(np.sqrt(pvs * (1 - pvs) / pss))
        
        stdev = np.sqrt(val * (1 - val) / nsamp) + 0.01 / WEIGHTS[grade] - stdev_shift
        
        data = 0, stdev, val
        self.data.add(data)
        
    def run_suite(self):
        self.suite.UpdateSet(self.data)
        
        
    def display(self):
        pmf = np.array(list(self.suite.Items()))
        plt.plot(pmf[:,0], pmf[:,1])
    
    def e_val(self):
        pmf = np.array(list(self.suite.Items()))
        return np.dot(pmf[:,0], pmf[:,1])

print('Libraries loaded')

districts = list(pd.read_csv('./data/district_input.csv').iloc[:,0])
poll_df = pd.read_csv('./data/poll_input.csv')

polls = {district: PollAggregator() for district in districts}
vanilla_weights = {district: [] for district in districts}

print("Poll computation ready")

for index, row in poll_df.iterrows():
    name = row['district_name']
    dem = row['dem_percent']
    repub = row['rep_percent']
    val = dem / (dem + repub)
    grade = row['pollster_grade']
    
    year = int(str(row['date']).split('/')[2])
    month = int(str(row['date']).split('/')[0])
    day = int(str(row['date']).split('/')[1])
    d0 = date(year, month, day)
    d1 = date(2018, 11, 6)
    days = d1 - d0
    days = days.days
    
    nw = np.exp(days / 30)
    nsamp = row['sample_size']
    
    polls[name].update(val, grade, nsamp, days)
    vanilla_weights[name].append(np.exp(days / 167) * WEIGHTS[grade])

print('Polls processed')

for d in polls:
    if len(polls[d].poll_vals) > 0:
        polls[d].run_suite()
        print('run' + d)

print('Bayesian model run')

for vw in vanilla_weights:
    if len(vanilla_weights[vw]) > 0:
        vanilla_weights[vw] = 1.8 / np.pi * np.arctan(16.6 * np.sum(vanilla_weights[vw]))
    else:
        vanilla_weights[vw] = 0
        
with open('./data/ppoll.csv', 'w') as f:
    for poll in polls:
        if len(polls[poll].poll_vals) > 0:
            f.write(poll + ',' + str((polls[poll].e_val() - 0.5) * 100) + '\n')
            polls[poll].display()
        else:
            f.write(poll + ',' + '0\n')

print("Polls written")            

pp = pd.read_csv('./data/ppoll.csv', header=None)
# bf = pd.read_csv('big_fun.csv')
df = pd.read_csv('./data/demographics.csv', header=None)

ins = ['S' + str(rep).zfill(3) for rep in range(df.shape[1])]
outs = ['MRAM']

# Drop rows that have non-numerical data
df.dropna(inplace=True)
df = df[df.applymap(np.isreal).any(1)]
df.columns = ins
df['Name'] = districts
df['Tmp'] = pp.iloc[:,1]

# Make a copy of all the data before you drop the districts without polls
raw = df.copy(deep=True)

# Drop rows that down have MRAM
# df['MRAM'] = (50 + df['Tmp']) / 100 - bf['Fund only']
df['MRAM'] = (50 + df['Tmp']) / 100 - 0.5
df = df[df['Tmp'] != 0]
df = df[df['Name'].str.get(0) + df['Name'].str.get(1) != 'PA']

print('Data loaded')

def gen_interpolator(df_train, _ins, _outs):
    inrep = [np.array(df_train[rep]).astype(float) for rep in _ins]
    outrep = [np.array(df_train[rep]).astype(float) for rep in _outs]
    features = list(inrep) + list(outrep)
    return scipy.interpolate.Rbf(*features)


def validate_step(rbfi, df_validate, _ins, _outs):
    y_pred = []
    y_true = []
    for index, row in df_validate.iterrows():
        y_pred.append(rbfi(*[row[rep] for rep in _ins]))
        y_true.append(row[outs[0]])

    er = np.array(np.array(y_true) - np.array(y_pred))
    correct = np.array([(y_pred[i] > 0.5) == (y_true[i] > 0.5) for i in range(len(y_pred))])
    
    return er, correct

totcor = 0
er_all = []
correct_all = []

print()
N = 100
for k in range(N):
    # Randomly select training 
    ridx = np.random.rand(len(df)) < 114 / 435 # 50% True 50% False
    df_train = df[ridx]
    df_validate = df[~ridx]
    
    rbfi = gen_interpolator(df_train, ins, outs)
    er, correct = validate_step(rbfi, df_validate, ins, outs)
    
    er_all.extend(er)
    correct_all.extend(correct)
    
    if k % (N / 10) == 0:
        print('- ' + str(k / N * 100) + '% -')

print()
print('\n\nFraction of Races Predicted Correctly: ' + str(np.sum(correct_all) / len(correct_all)))
print('Mean Squared Prediction Error: ' + str(np.mean(np.array(er_all) ** 2)))
print('Mean Absolute Prediction Error: ' + str(np.mean(np.abs(er_all))))
print('Mean Prediction Error: ' + str(np.mean(er_all)))
print('Stdev Prediction Error: ' + str(np.std(er_all)))

#plt.hist(er_all)
#plt.xlabel('Error')
#plt.ylabel('Frequency')
#plt.title('Prediction Error (Histogram)')
#plt.show()

#res = stats.probplot(er_all, plot=plt)
#plt.title('Prediction Error (Normal Probability Plot)')
#plt.show()

ers = []
names = []
for k in range(len(df)):
    ridx = np.ones(len(df), dtype=bool)
    ridx[k] = False
    
    df_train = df[ridx]
    df_validate = df[~ridx]
    
    rbfi = gen_interpolator(df_train, ins, outs)
    er, correct = validate_step(rbfi, df_validate, ins, outs)
    
    name = df_validate['Name'].iloc[0]

    ers.append(er)
    names.append(name)

vws = np.array([vanilla_weights[name] for name in names])
adjws = 1 / np.array(ers)

aw = np.average(vws, weights=adjws[:,0])
print('Interpolator weight: ' + str(aw))

rbfi = gen_interpolator(df, ins, outs)

y_pred = []
for index, row in raw.iterrows():
    interpout = rbfi(*[row[rep] for rep in ins])
#     y_pred.append(interpout + bf['Fund only'].iloc[index])
    y_pred.append(interpout + 0.5)

out_df = pd.DataFrame({'district_name': districts, 'bv': y_pred})
out_df.to_csv('bv_out.csv', index=False)

print(np.sum(np.array(y_pred) > .50) / len(y_pred))