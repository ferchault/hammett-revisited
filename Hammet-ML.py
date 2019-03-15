#!/usr/bin/env python
# coding: utf-8

# ## We compare the training of Hammett, Hammett + $\Delta$-ML, and ML. The goal is to see if and how Hammett improves the predictive power. For each training set size we use a 10-fold cross validation

import os
from functools import reduce

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import qml
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve

def Getsn2(basepath):
    ''' This function parses the txt file to extract the Activation Energies of the Sn2 reaction
        It takes only the lowest energy conformer and drops duplicates
        
        Parameters
        ----------
        basepath: string
            path to the folder containing the txt file

        Returns
        ----------
        sn2: pandas dataframe
            one row for each molecule with the corresponding activation energy

    
    
        '''
    
    df1 = pd.read_csv(basepath + '/electronic-forward-barriers.txt', sep=' ', header=None)
    
    df1.columns = ['reaction', 'folder', 'label', 'TSEnergy', 'rank', 'ConfID',               'dist', 'theory', 'ReactEnergy', 'rank2', 'Barrier']
    
    df1 = df1.query("rank2 == 1 & rank == 1")
    
    df1 = df1.drop(['folder','TSEnergy', 'ConfID', 'dist', 'theory', 'ReactEnergy'], axis=1)
    
    sn2 = df1.loc[(df1['reaction'] == 'sn2')].drop(['reaction', 'rank', 'rank2'], axis=1)
    
    sn2 = sn2.sort_values(by=['label', 'Barrier']).drop_duplicates(subset = 'label')
    
    sn2['Barrier'] = sn2['Barrier'].apply(lambda x:x*627.5095)
    
    sn2 = sn2.drop(sn2.loc[(abs(sn2['Barrier']) > 100)].index, axis=0)
    
    #sn2 = sn2.drop_duplicates(subset='label')
    
    sn2 = sn2.reset_index(drop=True)
    return(sn2)


# In[16]:


def reshapecopy(datah):
    '''This function reshapes the initial database to separate the subtituents (along the rows)
        from the reactions (one for each column)
        
        Parameters
        ----------
        data : Pandas DataFrame
            One row for each molecule with the corresponding Activation Energies and index
         
        Returns
        -----------
        df_final2 : Pandas DataFrame
            Each row is a set of substituent and each column a reaction. Contains the corresponding Index
    
    '''
    data = sn2.copy()
    data['reaction'] = data.label.str[-3:]    
    data['label'] = data['label'].str[0:-4]
    def _colname(current):
        if current == 'label':
            return current
        return 'DeltaG(%s)' % current.replace('_', '')
    
    q = data.pivot(index='label', columns='reaction', values='Barrier').reset_index()
    q.columns = list(map(_colname, q.columns))
    return q


def reshape4hamm(datah):
    '''This function reshapes the initial database to separate the subtituents (along the rows)
        from the reactions (one for each column)
        
        Parameters
        ----------
        datah : Pandas DataFrame
            One row for each molecule with the corresponding Activation Energy
         
        Returns
        -----------
        df_final2 : Pandas DataFrame
            Each row is a set of substituent and each column a reaction. Contains the correspondin Ea
    
    '''
    data = datah.copy()
    data['reaction'] = data.label.str[-3:]    
    data['label'] = data['label'].str[0:-4]
    
    AA = data.loc[(data['reaction'] == 'A_A')].drop(['reaction'], axis=1)
    AB = data.loc[(data['reaction'] == 'A_B')].drop(['reaction'], axis=1)
    AC = data.loc[(data['reaction'] == 'A_C')].drop(['reaction'], axis=1)
    AD = data.loc[(data['reaction'] == 'A_D')].drop(['reaction'], axis=1)

    BA = data.loc[(data['reaction'] == 'B_A')].drop(['reaction'], axis=1)
    BB = data.loc[(data['reaction'] == 'B_B')].drop(['reaction'], axis=1)
    BC = data.loc[(data['reaction'] == 'B_C')].drop(['reaction'], axis=1)
    BD = data.loc[(data['reaction'] == 'B_D')].drop(['reaction'], axis=1)

    CA = data.loc[(data['reaction'] == 'C_A')].drop(['reaction'], axis=1)
    CB = data.loc[(data['reaction'] == 'C_B')].drop(['reaction'], axis=1)
    CC = data.loc[(data['reaction'] == 'C_C')].drop(['reaction'], axis=1)
    CD = data.loc[(data['reaction'] == 'C_D')].drop(['reaction'], axis=1)
    
    dfs = [AA,AB,AC,AD,BA,BB,BC,BD,CA,CB,CC,CD]
    
    df_interm = reduce(lambda left,right: pd.merge(left,right, how = 'outer', left_on=['label'], right_on = ['label']), dfs)
    
    df_interm.columns = ['label','DeltaG(AA)','DeltaG(AB)','DeltaG(AC)','DeltaG(AD)','DeltaG(BA)','DeltaG(BB)','DeltaG(BC)',                     'DeltaG(BD)','DeltaG(CA)','DeltaG(CB)','DeltaG(CC)','DeltaG(CD)']
    
    df_final2 = df_interm.set_index(['label'])
    
    df_final2 = df_final2.dropna(axis=0, how='all')
    
    df_final2 = df_final2.sort_index(axis=0)
    
    df_final2 = df_final2.drop_duplicates()
    
    return(df_final2)


# In[9]:


def get_onehot_matr(data):
    ''' This function generates the one-hot encoding representation for all the molecules in the entire dataset
        and puts them all into one array
        
        Parameters
        ----------
        data : pandas DataFrame
            of all the molecules and the corresponding barriers (as output of getSn2)
    
        Returns
        ----------
        sigmas : 1D np.array
            with the one-hot encoding for each moelcule
        
    '''
    onepot = []
    for i in [ _.split('_') for _ in data['label'] ] :
        for pos in range(4):
            for lett in 'A B C D E'.split():
                onepot.append(i[pos] == lett)
    
        for x in 'A B C'.split():
            onepot.append(i[4] == x)
    
        for y in 'A B C D'.split():
            onepot.append(i[5] == y)
            
    onepot = np.array(onepot).reshape(len(data), 27).astype(np.int)
    
    return(onepot)


# In[11]:


def Hammett_data(data):
    '''This function calculates the rhos, sigmas and E0 of the Hammett regression. 
        
        Parameters
        ----------
        data : Pandas DataFrame
            One row for each molecule with the corresponding Activation Energy
         
        Returns
        -----------
        dicnewrho: dictionary
            key:= reaction, value:= rho value
            
        E0: 1D np.array
            with the 12 E0, one for each reaction
            
        dicsigmas: dictionary
            key:= R1_R2_R3_R4, value:= sigma
            
        rhos: 1D np.array
            of the same 12 rhos (same found in dictrhos)
       
    '''
    x0 = data.median(axis=0).values
    dicrho, rhos = calc_init_rho(data)
    dicsigmas, sigmas = calc_sigmas(data - x0, rhos) 
    dicnewrho, E0 = fix_param(data - x0, sigmas, dicrho)
    E0 = E0 + x0
    #dicE0 = dict(zip(combinations, E0))
    return(dicnewrho, E0, dicsigmas, rhos)


# In[12]:


def Hamm_eval(data, dicrho, dicsigmas, dicE0):
    '''This function predicts the Activation Energy using the Hammett regression
        
        Parameters
        ----------
        data : Pandas DataFrame
            One row for each molecule with the corresponding Activation Energy
            
        dicrho: dictionary
            key:= reaction, value:= rho value
         
        dicsigmas: dictionary
            key:= R1_R2_R3_R4, value:= sigma
            
        dicE0: dictionary
            key:= reaction, value:= E0 value
        
        Returns
        -----------
        pred: 1D np.array
            of the predicted Activation Energies 
       
    '''
    pred = []
    for index,row in data.iterrows():
        subst = row['label'][0:7]
        rxn = row['label'][8:9]+row['label'][10:11]
        pred.append( (dicrho[rxn]*dicsigmas[subst]) + dicE0[rxn])

    return(np.array(pred))


# In[13]:


def Hamm_param(traindf, testdf):
    ''' This function predicts the Activation Energy using the Hammett regression
        
        Parameters
        ----------
        traindf: pandas dataframe
            df of molecules and Activation Energies for the TRAINING set
        
        traindf: pandas dataframe
            df of molecules and Activation Energies for the TEST set
        
        Returns
        -----------
        dicrho: dictionary
            key:= reaction, value:= rho value
         
        dicsigmfin: dictionary
            key:= R1_R2_R3_R4, value:= sigma
            
        dicE0: dictionary
            key:= reaction, value:= E0 value
        
      
    '''

    dicrho, E0, dicsigmas, rhos = Hammett_data(reshape4hamm(traindf))
    dicsigmas_test = calc_sigmas(reshape4hamm(testdf) - E0,rhos)[0]
    dicsigmfin = {**dicsigmas_test, **dicsigmas}
    dicE0 = dict(zip(combinations, E0))
    return(dicrho, dicsigmfin, dicE0)


# In[8]:


def fix_param(deltadata, sigmas, dicrhos):
    """Modifies the Hammett parameter to obtain a better fit
        uses the difference between the existing data and the ideal correlation 
        in the correlation plots
    
        Parameters
        --------------
        deltadata : pandas dataFrame
            of deltaG over reactions and substituents shifted by the Ea0 (calculated by median or whatever)
    
        sigmas : 1D np.array
            set of the sigmas
    
        dicrhos :  dict
            key := reaction, value := rho
    
    
        Returns
        ---------------
        dicnewrho : dict
            contains the new values for rho
        
        newEa0 : 1D np.array
            New shift for Ea0, to be subtracted from the data df for Hammett to work
    
    
        """
    newrho = []
    newEa0 = []
    newdata = deltadata.copy()
    for i in combinations:
        varx = deltadata['DeltaG(%s)' %i].values
        vary = sigmas*dicrhos['%s' %i]
        mask = ~np.isnan(varx) & ~np.isnan(vary)
        slope, intercept = stats.linregress(varx[mask], vary[mask])[0:2]
        newrho.append(dicrhos['%s' %i]-slope+1)
        newEa0.append(-intercept)
    newrho = np.array(newrho)
    newEa0 = np.array(newEa0)
    dicnewrho = dict(zip(combinations, newrho))
    
    return(dicnewrho, newEa0)


# In[5]:


def calc_m(data):
    """ Create the A matrix for the overdetermined system of equations Ax=0
        in this matrix, each entry is weighted by the variance of m

        The output matrix has a column for each reaction XY and a row for each pair of reaction XY_X'Y'
        If XY == X'Y' the row is skipped (132 rows in total)

        This matrix is used to solve the system (Rho_X'Y' * m - Rho_XY = 0)
        
        Each entrance is weighted by the covariance of the linear regression

        Parameters
        ----------
        data: pandas DataFrame
            of deltaG over reactions and substituents

        Returns
        ----------
        m_matrix : 2D np.array
            matrix A for the WLSQ

        Other
        ----------
        This function is only called for a linealg.lstsq
    """
    m_matrix=np.zeros(12)
    for a,i in enumerate(combinations):
        for b,j in enumerate(combinations):
            if (i==j): continue
            pippo = data[['DeltaG(%s)' %i, 'DeltaG(%s)' %j]].copy().dropna(axis=0,how='any')
            pippo.columns = ['first', 'second']
            if (len(pippo)<2):
                continue
            elif (len(pippo)<5):
                newline=np.zeros(12)
                param = np.polyfit(pippo[pippo.columns[0]].values,pippo[pippo.columns[1]].values,1)
                newline[a]= (-1)
                newline[b]= param[0]
                m_matrix = np.vstack((m_matrix,newline))
            else:
                newline=np.zeros(12)
                param,cov = np.polyfit(pippo[pippo.columns[0]].values,pippo[pippo.columns[1]].values,1,cov=True)
                newline[a]= (-1/cov[0,0])
                newline[b]= param[0]/cov[0,0]
                m_matrix = np.vstack((m_matrix,newline))
    m_matrix=np.delete(m_matrix, 0, 0)

    
    return(m_matrix)


# In[6]:


def calc_init_rho(data):
    """ This function generates the INITIAL set of rhos (BEFORE the eventual Self-Consistency)
        WLSQ regression on the system Ax=b, where we fix the fisrt m to be 1 to avoid trivial solutions
        For this reason the b vector is not only zeros, but contains the first column of the matrix, which has been
        dropped from the A matrix
    
        Parameters
        ----------
        data: pandas DataFrame
            of deltaG over reactions and substituents
    
        Returns
        ----------
        dicrho : dict
            assigns to each reaction the corresponding value of rho
    
        array(Wlsqrho) : 1D np.array
            contains the values of rhos
    """
    
    rhoAA = np.array([1])
    
    slopesmatr = calc_m(data)
    
    regr_rhos = scipy.sparse.linalg.lsmr(slopesmatr[:,1:] , -slopesmatr[:,0])[0]
    
    rhos = np.append(rhoAA, regr_rhos)
    
    dicrho = dict(zip(combinations, rhos))
    
    return(dicrho, rhos)


# In[7]:


def calc_sigmas(data, rhos):
    """ This function generates the set of sigmas
    
        Parameters
        ----------
        data : pandas DataFrame
            of deltaG over reactions and substituents
    
        dicrhos :  dict
            key := reaction, value := rho
    
        Returns
        ----------
        dicsigmas :  dict
            key := string R_1 to R_4, value := sigma
        
        sigmas : 1D np.array
            set of the sigmas
    """

    sigmas = (data / rhos).mean(axis=1).values
    
    dicsigmas = dict(zip(data.index.values, sigmas))
    
    return(dicsigmas, sigmas)


# In[15]:


def sliceKernels(traindf, testdf, Kernel):
    '''This function predicts the Activation Energy using the Hammett regression
        
        Parameters
        ----------
        traindf: pandas dataframe
            df of molecules and Activation Energies for the TRAINING set
        
        traindf: pandas dataframe
            df of molecules and Activation Energies for the TEST set
        
        Kernel: 2D np.array
            matrix with the complete Kernel
       
        Returns
        -----------
        trainKernel: 2D np.array
            matrix with the kernel to be used for the training
            
        testKernel: 2D np.array
            matrix with the kernel to be used for the validation
       
    '''
    trainKernel = Kernel[np.ix_(list(traindf.index),list(traindf.index))]
    testKernel = Kernel[np.ix_(list(testdf.index),list(traindf.index))]
    return(trainKernel, testKernel)


# In[14]:


def KRR_pred(Y_train, trainKernel, testKernel):
    '''This function predicts the Activation Energy using the Hammett regression
        
        Parameters
        ----------
        Y_train: 1D np.array
            array with the Y observables used for the trainig
        
        trainKernel: 2D np.array
            matrix with the kernel to be used for the training
            
        testKernel: 2D np.array
            matrix with the kernel to be used for the validation
       
        Returns
        -----------
        pred: 1D np.array
            of the predicted observables 
       
    '''
    alpha = cho_solve(trainKernel, Y_train)
    pred = np.dot(testKernel, alpha)
    return(pred)


# In[17]:


def traintest(data, TSsize):
    ''' This functions selects the entries on the trainig set such that it is always possible to calculate the rhos
        of the system. To do so, it makes sure that at least one overlap cell for each column/row has at least
        5 entries
    
        Parameters
        ----------
        data: Pandas DataFrame
            One row for each molecule with the corresponding Activation Energies and index
         
        TSsize: int
            size of the training set
        
        Returns
        -----------
        traindf: pandas dataframe
            df of molecules and Activation Energies for the TRAINING set
        
        traindf: pandas dataframe
            df of molecules and Activation Energies for the TEST set
        '''
    sncopy = data.copy()
    sncopy['idx'] = sncopy.index
    rescopy = reshapecopy(sncopy)
    
    
    while True:
        eyematr = np.eye(12)
        np.random.shuffle(eyematr)
        if ( np.sum(np.diag(eyematr)) == 0 ):
            break

    tomove = np.argwhere(eyematr == 1)

    moveidx = []

    for i in tomove:
        #tmpdf = data[[data.columns[i[0]], data.columns[i[1]]]].dropna(axis=0,how='any')
        tmpdf = rescopy[[rescopy.columns[i[0]], rescopy.columns[i[1]]]].dropna(axis=0,how='any')

        moveidx = moveidx + tmpdf.sample(5).values.ravel().tolist()


    moveidx = list(set(moveidx))

    moveidx = [ int(_) for _ in moveidx]


    traindf = sncopy.iloc[moveidx]
    shuffrest = sncopy.drop(index=moveidx).sample(frac=1)
    part2 = shuffrest.head(TSsize-len(sncopy.iloc[moveidx]))

    traindf = traindf.append(part2)
    traindf = traindf.drop(['idx'], axis=1)

    testdf = shuffrest.tail(-(TSsize-len(sncopy.iloc[moveidx])))
    testdf = testdf.drop(['idx'], axis=1)
    
    return(traindf,testdf)


if __name__ == '__main__':
    basepath=os.getcwd() + '/data'

    combinations='AA AB AC AD BA BB BC BD CA CB CC CD'.split( )

    sn2 = Getsn2(basepath)

    onehot = get_onehot_matr(sn2)
    KernelML = laplacian_kernel(onehot, onehot, 3000)
    KernelML[np.diag_indices_from(KernelML)] += 1e-8

    KernelDML = laplacian_kernel(onehot, onehot, 23000)
    KernelDML[np.diag_indices_from(KernelDML)] += 1e-8


    er_hamm_ls = []
    er_dml_ls = []
    er_ml_ls =[]

    for TSsize in np.arange(200,1301,100):
        hammae = []
        dmlmae = []
        mlmae = []
        
        for fold in range(15):
            traindf, testdf = traintest(sn2, TSsize)

            MLtrainKernel, MLtestKernel = sliceKernels(traindf, testdf, KernelML)
            DMLtrainKernel, DMLtestKernel = sliceKernels(traindf, testdf, KernelDML)
                
            dicrho, dicsigmas, dicE0 = Hamm_param(traindf, testdf)

            Ham_prediction = Hamm_eval(testdf, dicrho, dicsigmas, dicE0)
            Hamm_residuals = testdf['Barrier'].values -  Ham_prediction

            Ham_mae =  np.mean(np.abs( Hamm_residuals  ))
            #print("     Hamm MAE = ", Ham_mae)
            hammae.append(Ham_mae)

            DMLY_train = traindf['Barrier'].values - Hamm_eval(traindf, dicrho, dicsigmas, dicE0)
            DMLY_pred = KRR_pred(DMLY_train, DMLtrainKernel, DMLtestKernel)
            DMLY_mae = np.mean(np.abs( DMLY_pred - Hamm_residuals  ))
            #print("     DML MAE = ", DMLY_mae)
            dmlmae.append(DMLY_mae)

            MLY_train = traindf['Barrier'].values
            MLY_test = testdf['Barrier'].values
            MLY_pred = KRR_pred(MLY_train, MLtrainKernel, MLtestKernel)
            MLY_mae = np.mean(np.abs( MLY_pred - MLY_test  ))
            #print("     ML MAE = ", MLY_mae)
            mlmae.append(MLY_mae)
            
        
        #er_ml1.append(np.mean(np.array(ML_err)))
        er_hamm_ls.append(np.mean(np.array(hammae)))
        er_dml_ls.append(np.mean(np.array(dmlmae)))
        er_ml_ls.append(np.mean(np.array(mlmae)))
       


    # In[20]:


    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #get_ipython().run_line_magic('matplotlib', 'notebook')
    plt.plot(np.arange(200,1301,100), np.array(er_hamm_ls), '--o', label='Hammett')
    plt.plot(np.arange(200,1301,100), np.array(er_dml_ls), '--o', label=r'$\Delta$-ML')
    plt.plot(np.arange(200,1301,100), np.array(er_ml_ls), '--o', label=r'ML')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    plt.xticks(np.array([200,500,800,1000, 1200]), np.array([200,500,800,1000, 1200]), rotation=45)
    plt.yticks(np.array([2.5,3,4,5,6]), np.array([2.5,3,4,5,6]))
    plt.xlabel('Training set size')
    plt.ylabel('MAE (kcal/mol)')
    plt.title('One-hot encoding learning curves')
    plt.grid(True)
    plt.show()
    plt.savefig("ugly-learning-curves.png", dpi=500, format="png")


