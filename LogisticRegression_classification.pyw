import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
LR_log = open('LR_log.log', mode = 'a',encoding='utf-8')
for i in range(10):
    print("sdjlahjljag", file=LR_log)
LR_log.close()

train_df = pd.read_csv(r'input\train_ dataset\nCoV_100k_train.labled.csv',engine ='python')
test_df  = pd.read_csv(r'input\test_dataset\nCov_10k_test.csv',engine ='python')

train_df = train_df[train_df['情感倾向'].isin(['0','1','-1'])]
train_df['time'] = pd.to_datetime('2020年' + train_df['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')

train_df['month'] =  train_df['time'].dt.month
train_df['day'] =  train_df['time'].dt.day
train_df['dayfromzero']  = (train_df['month']-1)*31 +  train_df['day']

train_df['weibo_len'] = train_df['微博中文内容'].astype(str).apply(len)

train_df['pic_len'] = train_df['微博图片'].apply(lambda x: len(eval(x)))
'''#'''

train_df['text_cut'] = train_df['微博中文内容'].apply(lambda x:" ".join(jieba.cut(str(x))))
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_df['text_cut'])
xtrain_count =  count_vect.transform(train_df['text_cut'])
lr = LogisticRegression(verbose = 1)
lr.fit(xtrain_count, train_df['情感倾向'] )

test_df['text_cut'] = test_df['微博中文内容'].apply(lambda x:" ".join(jieba.cut(str(x))))
xtest_count =  count_vect.transform(test_df['text_cut'])
test_sub_lr = lr.predict(xtest_count)
print(test_sub_lr)

test_sub_1 = []
for i in test_sub_lr:
    if i == '-1':
        test_sub_1.append("--消极情绪")
    elif i == '0':
        test_sub_1.append("--中立情绪")
    elif i =='1':
        test_sub_1.append("--积极情绪")

a = test_df['微博中文内容']
for i in range(len(test_sub_1)):
    if i % 5 ==0:
        print(a[i],"----",test_sub_1[i])


c={"微博中文内容":a,
   '情感倾向':test_sub_1}

out_data= pd.DataFrame(c)
# print(out_data)
out_data.to_csv('lr_out_put.csv')

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit
from joblib import Parallel, delayed, effective_n_jobs

class LinearClassifierMixin():
    """Mixin for linear classifiers.

    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):

        check_is_fitted(self)

        X = check_array(X, accept_sparse='csr')

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """
        预测X中样本的类别标签。

        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        """Logistic回归的概率估计。

        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

class SparseCoefMixin:
    """Mixin for converting coef_ to and from CSR format.

    L1-regularizing estimators should inherit this.
    """

    def densify(self):
        """
        Convert coefficient matrix to dense array format.

        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.

        Returns
        -------
        self
            Fitted estimator.
        """
        msg = "Estimator, %(name)s, must be fitted before densifying."
        check_is_fitted(self, msg=msg)
        if sp.issparse(self.coef_):
            self.coef_ = self.coef_.toarray()
        return self

    def sparsify(self):
        """
        Convert coefficient matrix to sparse format.

        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.

        The ``intercept_`` member is not converted.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.

        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.
        """
        msg = "Estimator, %(name)s, must be fitted before sparsifying."
        check_is_fitted(self, msg=msg)
        self.coef_ = sp.csr_matrix(self.coef_)
        return self

