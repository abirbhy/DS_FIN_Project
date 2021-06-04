########################################################################################################################
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
import seaborn as sns
from flask import Flask, render_template, request
from matplotlib import pyplot
from xgboost import plot_importance, XGBClassifier

dataset = pd.read_excel('./Dataset.xlsx', sheet_name='data', engine='openpyxl')
#print(dataset.dtypes)
#print(dataset.info())
#print(dataset.describe())

########################################################################################################################
# Handling missing values

# get the number of missing data points per column
missing_values_count = dataset.isnull().sum()

# how many total missing values do we have?
total_cells = np.product(dataset.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
#print(percent_missing)

##Finding the the count and percentage of values that are missing in the dataframe.
df_null = pd.DataFrame({'Count': dataset.isnull().sum(), 'Percent': 100*dataset.isnull().sum()/len(dataset)})

##printing columns with null count more than 0
#print(df_null[df_null['Count'] > 0][:])

#print(dataset.X4.value_counts())
#print(dataset.X7.value_counts())
#print(dataset.X9.value_counts())
dataset['X7'] = dataset['X7'].fillna(dataset['X7'].value_counts().index[0])
dataset['X9'] = dataset['X9'].fillna(dataset['X9'].value_counts().index[0])
dataset['X4'] = dataset['X4'].fillna(dataset['X4'].mean())
##Finding the the count and percentage of values that are missing in the dataframe.
df_null = pd.DataFrame({'Count': dataset.isnull().sum(), 'Percent': 100*dataset.isnull().sum()/len(dataset)})
#print(df_null[df_null['Count'] > 0][:])
#dataset['Montant'] = dataset['Montant'].fillna(dataset['Montant'].mean())

#for col in dataset.columns:
#    print(dataset[col].value_counts())
##printing columns with null count more than 0
#print(df_null[df_null['Count'] > 0][:])

########################################################################################################################
# Converting categorical to numerical values

for col in ['X1', 'X3', 'X5', 'X10']:
    dataset[col] = dataset[col].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns
dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
#print(dataset.dtypes)

#print(dataset)

########################################################################################################################
# Correlation heatmap

def correlation_heatmap(data):
  plt.figure(figsize=(16, 6))
  heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
  heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
  # save heatmap as .png file
  # dpi - sets the resolution of the saved image in dots/inches
  # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
  plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
  plt.show()
  return

#correlation_heatmap(dataset)

def correl(X_train):
    cor = X_train.corr()
    corrm = np.corrcoef(X_train.transpose())
    corr = corrm - np.diagflat(corrm.diagonal())
    #print("max corr:",corr.max(), ", min corr: ", corr.min())
    c1 = cor.stack().sort_values(ascending=False).drop_duplicates()
    high_cor = c1[c1.values!=1]
    ## change this value to get more correlation results
    thresh = 0.9
    #print(high_cor[high_cor>thresh])
correl(dataset)
# drop X11 from dataset because it presents high correlation
del(dataset['X11'])
#correlation_heatmap(dataset)
correl(dataset)

########################################################################################################################
# Feature selection

X = dataset.loc[:, dataset.columns != 'Target']
y = dataset['Target']

# plot feature importance manually

model = XGBClassifier()
model.fit(X, y)
# feature importance
#print(model.feature_importances_)
# plot feature importance
plot_importance(model)
pyplot.show()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # split into inputs and outputs
    l = []
    for x in request.form.values():
        #print(x)
        partitioned_string = x.rpartition('=')
        #print(partitioned_string)
        l.append(partitioned_string[0])
    #print(l)

    if l[4] == '' and l[5] == '':
        l.remove(l[5])
        l.remove(l[4])
    elif l[4] != '' and l[5] == '':
        l.remove(l[5])
    #print(l)
    n = len(l)
    if n == 4:
        X = dataset[[l[0], l[1], l[2], l[3]]]
    elif n == 5:
        X = dataset[[l[0], l[1], l[2], l[3], l[4]]]
    elif n == 6:
        X = dataset[[l[0], l[1], l[2], l[3], l[4], l[5]]]
    y = dataset['Target']
    #print(X.shape, y.shape)

    # split into train test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    def run_exps(X, y):
        '''
        Lightweight script to test many models and find winners
    :param X_train: training split
        :param y_train: training target vector
        :param X_test: test split
        :param y_test: test target vector
        :return: DataFrame of predictions
        '''

        dfs = []

        models = [
            ('RF', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier()),
            ('SVM', SVC()),
            ('GNB', GaussianNB())
        ]
        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        target_names = ['0', '1']
        res = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
            cv_results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
            clf = model.fit(X, y)
            y_pred = clf.predict(X)

            liste = []
            liste2 = []
            for x in request.form.values():
                # print(x)
                partitioned_string = x.rpartition('=')
                # print(partitioned_string)
                liste.append(partitioned_string[2])
            # print(liste)

            if liste[4] == '' and liste[5] == '':
                liste.remove(liste[5])
                liste.remove(liste[4])
            elif liste[4] != '' and liste[5] == '':
                liste.remove(liste[5])
            for i in range(len(liste)):
                liste2.append(float(liste[i]))
            # print(liste2)

            Xnew = [liste2]
            # Xnew = [[1, 46, 0, 26, 0, 2, 1, 0, 3, 0, 900000, 221, 8161, 1084000, 13593, 1054170, 0.6003, 77.5524, 0.8303]]

            res.append(name)
            res.append(model.predict(Xnew)[0])
            res.append(';')

            print(name)
            print(classification_report(y, y_pred, target_names=target_names))
            results.append(cv_results)
            names.append(name)
            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)
            final = pd.concat(dfs, ignore_index=True)
        res.pop()
        print(res)
        bootstraps = []
        for model in list(set(final.model.values)):
            model_df = final.loc[final.model == model]
            bootstrap = model_df.sample(n=30, replace=True)
            bootstraps.append(bootstrap)

        bootstrap_df = pd.concat(bootstraps, ignore_index=True)
        results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')
        time_metrics = ['fit_time', 'score_time']  # fit time metrics

        ## PERFORMANCE METRICS
        results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)]  # get df without fit data
        results_long_nofit = results_long_nofit.sort_values(by='values')

        ## TIME METRICS
        results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)]  # df with fit data
        results_long_fit = results_long_fit.sort_values(by='values')

        plt.figure(figsize=(12, 6))
        sns.set(font_scale=2.5)
        g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Comparison of Model by Classification Metric')
        plt.savefig('./benchmark_models_performance.png', dpi=300)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.set(font_scale=2.5)
        g2 = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Comparison of Model by Fit and Score Time')
        plt.savefig('./benchmark_models_time.png', dpi=300)
        plt.show()

        return render_template('index.html', prediction_text='The predicted target is: {}'.format(res))

    return run_exps(X, y)

if __name__ == "__main__":
    app.run(debug=True)
