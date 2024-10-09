import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Function to split data
def split_data(df, Req_Columns_clean, target_column, stratify=None):
    X = df[Req_Columns_clean]
    Y = df[target_column]
    return train_test_split(X, Y, test_size=0.3, random_state=1, stratify=stratify)

# Function to balance data
def balance_data(X_train, Y_train):
    smote = SMOTE(random_state=1)
    X_train, Y_train = smote.fit_resample(X_train, Y_train)
    return X_train, Y_train

# Function to create dummy variables
def create_dummies(X_train, X_test):
    # Concatenate the train and test data
    
    X_full = pd.concat([X_train, X_test])

    bool_cols = [col for col in X_full if X_full[col].nunique() == 2 and (X_full[col].dtype == 'object' or X_full[col].dtype == 'bool')]
    category_cols = [col for col in X_full if X_full[col].nunique() > 2 and X_full[col].dtype == 'object']

    # Dummies for categorical variables with more than 2 classes
    X_full = pd.get_dummies(X_full, columns=category_cols, drop_first=False)

    # Dummies for categorical variables with 2 classes
    X_full = pd.get_dummies(X_full, columns=bool_cols, drop_first=True)

    # Split the data back into train and test
    X_train = X_full.iloc[:X_train.shape[0], :]
    X_test = X_full.iloc[X_train.shape[0]:, :]

    return X_train, X_test

# Function to encode labels
def label_encoder(Y_train, Y_test):
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)
    return Y_train, Y_test

#creamos una funcion para entrenar los modelos utilizando algoritmos geneticos

def train_models(X_train, Y_train, param_grid, estimator):
    
    param_grid = param_grid
    
    evolved_estimator = GASearchCV( estimator = estimator,
                                    cv=10,
                                    scoring= 'balanced_accuracy',
                                    population_size=30,
                                    generations=15,
                                    elitism=True,
                                    crossover_probability=0.4,
                                    mutation_probability=0.6,
                                    param_grid=param_grid,
                                    criteria='max',
                                    n_jobs=-1,
                                    verbose=True)
    
    evolved_estimator.fit(X_train, Y_train)
    
    hiperparametros = evolved_estimator.best_params_
    bestmodel = evolved_estimator.best_estimator_
    
    return hiperparametros, bestmodel


#creamos una funcion para evaluar el modelo
def evaluate_model(X_test, Y_test, model):
    
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    
    Y_pred = model.predict(X_test)
    
    cm=metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
    
    #display the confusion matrix
    #disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=encoder.classes_)
    #disp.plot()
    
    epsilon = 1e-9
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn+epsilon)
    specificity = tn/(tn+fp + epsilon)
    balanced_accuracy = (sensitivity+specificity)/2
    f1_score = 2*tp/(2*tp+fp+fn + epsilon)
    precision = tp/(tp+fp + epsilon)
    false_positive_rate = fp/(fp+tn + epsilon)
    false_negative_rate = fn/(fn+tp + epsilon)
    positive_likelihood_ratio = sensitivity/false_positive_rate 
    negative_likelihood_ratio = false_negative_rate/specificity
    diagnostic_odds_ratio = positive_likelihood_ratio/negative_likelihood_ratio
    roc_auc = metrics.roc_auc_score(Y_test, Y_pred, multi_class='ovo')
    
    return {
            'Verdaderos positivos': tp,
            'Verdaderos negativos': tn,
            'Falsos positivos': fp,
            'Falsos negativos': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'positive_likelihood_ratio': positive_likelihood_ratio,
            'negative_likelihood_ratio': negative_likelihood_ratio,
            'diagnostic_odds_ratio': diagnostic_odds_ratio,
            'roc_auc': roc_auc
            }
