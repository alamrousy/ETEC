import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import matplotlib.pyplot as plt
# from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import tree
# from IPython import display

# class ScrollableWindow(QtWidgets.QMainWindow):
#     def __init__(self, fig):
#         self.qapp = QtWidgets.QApplication([])
# 
#         QtWidgets.QMainWindow.__init__(self)
#         self.widget = QtWidgets.QWidget()
#         self.setCentralWidget(self.widget)
#         self.widget.setLayout(QtWidgets.QVBoxLayout())
#         self.widget.layout().setContentsMargins(0, 0, 0, 0)
#         self.widget.layout().setSpacing(1)
#         # self.setGeometry(1000,1000)
#         # self.size
# 
#         self.fig = fig
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.draw()
#         self.scroll = QtWidgets.QScrollArea(self.widget)
#         self.scroll.setWidget(self.canvas)
# 
#         self.nav = NavigationToolbar(self.canvas, self.widget)
#         self.widget.layout().addWidget(self.nav)
#         self.widget.layout().addWidget(self.scroll)
# 
#         self.show()
#         exit(self.qapp.exec_())

class ETEC_PCA:
    def train(x_train,x_test,ETEC_n_components=None, *, ETEC_copy=True, ETEC_whiten=False, ETEC_svd_solver='auto', ETEC_tol=0.0, ETEC_iterated_power='auto',
             ETEC_random_state=None):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=ETEC_n_components,  copy=ETEC_copy, whiten=ETEC_whiten, svd_solver=ETEC_svd_solver, tol=ETEC_tol, iterated_power=ETEC_iterated_power,
                   random_state=ETEC_random_state)

        x_train=pca.fit_transform(x_train)
        x_test=pca.transform(x_test)

        return x_train,x_test


class ETEC_LDA:
    def train(x,y,x_test,ETEC_solver='svd', ETEC_shrinkage=None, ETEC_priors=None, ETEC_n_components=None, ETEC_store_covariance=False, ETEC_tol=0.0001, ETEC_covariance_estimator=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        LDA = LinearDiscriminantAnalysis(solver=ETEC_solver, shrinkage=ETEC_shrinkage, priors=ETEC_priors, n_components=ETEC_n_components, store_covariance=ETEC_store_covariance, tol=ETEC_tol, covariance_estimator=ETEC_covariance_estimator)

        lda_t = LDA.fit_transform(x,y)


        return LDA,lda_t



class ETEC_DATA:



    def Esplit(data,target,E_test_size):
        Y = data[target]
        data.drop(target, axis=1,inplace=True)
        try:
          X = data.convert_dtypes(convert_string=False)
          X = pd.get_dummies(X)
        except:
          ''

        X = X.fillna(0)
        X = X.astype(float)
        # X.to_excel('hoh.xlsx')
        scaler = StandardScaler()
        scaling = scaler.fit(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=E_test_size, random_state=0)

        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
        return X_train, X_test, Y_train, Y_test


    def Eclean(df):
        df_obj = df.select_dtypes(['object'])
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.lower())

        df.drop_duplicates(inplace=True)
        #remove ID
        # for i in df.columns:
        #     if len(df[i].unique()) >= len(df):
        #         df.drop(i, axis=1, inplace=True)
        #remove column contains more null,zero
        #print(df)
        for i in df.columns:
            countNull = sum(pd.isnull(df[i]))
            countZero = (df[i] == 0).sum()

            if (countNull / len(df) > 0.3 or countZero / len(df) > 0.5):
                df.drop(i, axis=1, inplace=True)
                continue
#remove column contains 70% of data
            for t in df[i].value_counts():
                if t / len(df) > 0.7:
                    df.drop(i, axis=1, inplace=True)
                    continue
        #print(df.shape)
        #df.to_excel('hoh.xlsx')
        #print(df)
        return df




    def Eopen(file, ETEC_Xsheet_name=0, ETEC_Xheader=0, ETEC_index_col=None, ETEC_engine=None,
             ETEC_skiprows=None, ETEC_nrows=None, ETEC_na_values=None,
             ETEC_C_header='infer'):
        file_name, file_extension = os.path.splitext(file)

        if file_extension == '.xlsx':
            df = pd.read_excel(file, sheet_name=ETEC_Xsheet_name, header=ETEC_Xheader, index_col=ETEC_index_col,
                               engine=ETEC_engine, skiprows=ETEC_skiprows, nrows=ETEC_nrows, na_values=ETEC_na_values)

        elif (file_extension == '.csv'):
            df = pd.read_csv(file, encoding='utf-8', engine=ETEC_engine, index_col=ETEC_index_col, header=ETEC_C_header,
                             skiprows=ETEC_skiprows, na_values=ETEC_na_values, nrows=ETEC_nrows)

        #if target in df:
            # print('exist')
         #   return df

       # else:
           # print('column is not exit in file')
        return df


    def EVisualization_Summary(df):

        temp_df = pd.DataFrame()
        for yy in df.columns:
            if df[yy].dtype == np.object:
                if len(pd.unique(df[yy])) < (len(df) * .1):
                    temp_df[yy] = df[yy]
                    temp_df[yy].fillna('NAN', inplace=True)

        ta, j = 0, 0
        PLOTS_PER_ROW = 2
        fig, axs = plt.subplots(math.ceil(len(temp_df.columns) / PLOTS_PER_ROW), PLOTS_PER_ROW, figsize=((len(temp_df.values)/(len(temp_df.columns.values))/4.7),((len(temp_df.values)/(len(temp_df.columns.values)))/4)),
                                squeeze=False, constrained_layout=True,
                    )

        col_map = plt.get_cmap('tab20b')

        for col in temp_df.columns:
            try:
                axs[ta][j].barh(temp_df[col].value_counts().index, temp_df[col].value_counts().values,
                                     color=col_map.colors, )
                # axs[ta][j].set_ylabel(col)
                axs[ta][j].set_title(col)

                j += 1
                if j % PLOTS_PER_ROW == 0:
                    ta += 1
                    j = 0

            except:
                continue

        # ScrollableWindow(fig)
        # plt.show()
        return fig


class ETEC_Decission_Tree:
    # def Draw(df, ETEC_DT):
    #     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10), dpi=1000)
    #     # for index in range(0, 2):
    #     #     tree.plot_tree(ETEC_RF.estimators_[index],
    #     #                    rounded=True,
    #     #                    filled=True,
    #     #                    ax=axes[index]);
    #     #
    #     #     axes[index].set_title('Estimator: ' + str(index), fontsize=5)
    #     fig = plt.figure(figsize=(50,50))
    #     _ = tree.plot_tree(ETEC_DT, filled=True,
    #                        dpi=50,
    #                        rounded=True,
    #                        fontsize=3, )
    #     ScrollableWindow(fig)
    #     return plt



    def Train(ETEC_X_Train,ETEC_Y_Train, *, ETEC_criterion='gini', ETEC_splitter='best',
              ETEC_max_depth=None, ETEC_min_samples_split=2, ETEC_min_samples_leaf=1,
              ETEC_min_weight_fraction_leaf=0.0, ETEC_max_features=None, ETEC_random_state=None,
              ETEC_max_leaf_nodes=None,
              ETEC_min_impurity_decrease=0.0, ETEC_class_weight=None, ETEC_ccp_alpha=0.0):
        from sklearn.tree import DecisionTreeClassifier

        ETEC_DT = DecisionTreeClassifier(criterion=ETEC_criterion, splitter=ETEC_splitter, max_depth=ETEC_max_depth,
                                         min_samples_split=ETEC_min_samples_split,
                                         min_samples_leaf=ETEC_min_samples_leaf,
                                         min_weight_fraction_leaf=ETEC_min_weight_fraction_leaf,
                                         max_features=ETEC_max_features, random_state=ETEC_random_state,
                                         max_leaf_nodes=ETEC_max_leaf_nodes,
                                         min_impurity_decrease=ETEC_min_impurity_decrease,
                                         class_weight=ETEC_class_weight, ccp_alpha=ETEC_ccp_alpha)
        ETEC_DT.fit(ETEC_X_Train, ETEC_Y_Train)

        return ETEC_DT
    def Prediction(ETEC_DT,ETEX_X_Test):
        # prediction
        y_pred = ETEC_DT.predict(ETEX_X_Test)
        #print("Predicted values:")
        return y_pred
        # print(y_pred)
        # accuracy
    def Confusion_metrix(ETEC_Y_test,y_pred):
        return confusion_matrix(ETEC_Y_test, y_pred)
    def Accuracy(ETEC_Y_test,y_pred):

        return accuracy_score(ETEC_Y_test, y_pred) * 100
    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df

    def Visualization(ETEC_Y_test,y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("decission tree", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt


class ETEC_Random_Forest:

    # def Draw(df, ETEC_RF):
    #     from sklearn.tree import plot_tree
    #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=1000)
    #     tree.plot_tree(ETEC_RF.estimators_[0],
    #
    #                    filled=True);
    #     # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 8), dpi=1000)
    #     # for index in range(0, 5):
    #     #     tree.plot_tree(ETEC_RF.estimators_[index],
    #     #                    precision=2,
    #     #                    filled=True,
    #     #                    ax=axes[index]);
    #     #
    #     #     axes[index].set_title('Estimator: ' + str(index), fontsize=10)
    #     # fig.savefig('imagename.png')
    #     # ScrollableWindow(fig)
    #     return fig

        #plt.show()

    def Train(ETEC_X_Train, ETEC_Y_Train, *, ETEC_n_estimators=100, ETEC_criterion='gini',
              ETEC_max_depth=None, ETEC_min_samples_split=2,
              ETEC_min_weight_fraction_leaf=0.0, ETEC_min_samples_leaf=1, ETEC_max_features='sqrt',
              ETEC_max_leaf_nodes=None,
              ETEC_min_impurity_decrease=0.0, ETEC_bootstrap=True, ETEC_oob_score=False, ETEC_n_jobs=None,
              ETEC_random_state=None, ETEC_verbose=0, ETEC_warm_start=False, ETEC_class_weight=None, ETEC_ccp_alpha=0.0,
              ETEC_max_samples=None):
        from sklearn.ensemble import RandomForestClassifier

        ETEC_RF = RandomForestClassifier(n_estimators=ETEC_n_estimators, criterion=ETEC_criterion,
                                         max_depth=ETEC_max_depth, min_samples_split=ETEC_min_samples_split,
                                         min_weight_fraction_leaf=ETEC_min_weight_fraction_leaf,
                                         max_features=ETEC_max_features, max_leaf_nodes=ETEC_max_leaf_nodes,
                                         min_impurity_decrease=ETEC_min_impurity_decrease, bootstrap=ETEC_bootstrap,
                                         oob_score=ETEC_oob_score,
                                         n_jobs=ETEC_n_jobs, random_state=ETEC_random_state, verbose=ETEC_verbose,
                                         warm_start=ETEC_warm_start,
                                         class_weight=ETEC_class_weight, ccp_alpha=ETEC_ccp_alpha,
                                         min_samples_leaf=ETEC_min_samples_leaf, max_samples=ETEC_max_samples)
        ETEC_RF.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return ETEC_RF
    def Prediction(ETEC_RF, ETEX_X_Test):

        return ETEC_RF.predict(ETEX_X_Test)
    def Confusion_metrix(ETEC_Y_test, y_pred):

        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df
#        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)


    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("Random Forest Classifier", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt



class ETEC_SVM:


    #def Draw(svc, X, y):
        # from sklearn.decomposition import PCA
        # from sklearn.svm import SVC
        #
        # from mlxtend.plotting import plot_decision_regions
        # import matplotlib.gridspec as gridspec
        # import itertools
        # gs = gridspec.GridSpec(2, 2)
        # # X=np.array()
        # pca = PCA()
        # X = pca.fit_transform(X)
        # y=np.array(y)
        # fig = plt.figure(figsize=(10, 8))
        # ETEC_SVM = SVC()
        # labels = [ 'SVM']
        # for clf, lab, grd in zip([ETEC_SVM],
        #                          labels,
        #                          itertools.product([0, 1], repeat=2)):
        #     clf.fit(X, y)
        #     ax = plt.subplot(gs[grd[0], grd[1]])
        #     fig = plot_decision_regions(X=X, y=y, clf=clf)
        #     plt.title(lab)

        # plt.show()

            # x_min, x_max = X[:, -1].min() - pad, X[:, -1].max() + pad
            # y_min, y_max = X[:, -1].min() - pad, X[:, -1].max() + pad
            # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
            # Z = Z.reshape(xx.shape)
            # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
            #
            # plt.scatter(X[:, 0], X[:, 1], s=70, c=y, cmap=mpl.cm.Paired)
            # # Support vectors indicated in plot by vertical lines
            # sv = svc.support_vectors_
            # plt.scatter(sv[:, 0], sv[:, 1], c='k', marker='x', s=100, linewidths='1')
            # plt.xlim(x_min, x_max)
            # plt.ylim(y_min, y_max)
            # plt.xlabel('X1')
            # plt.ylabel('X2')
            # plt.show()
            # print('Number of support vectors: ', svc.support_.size)
        # from mlxtend.plotting import plot_decision_regions
        #
        # from sklearn.decomposition import PCA
        #
        # clf = SVC( C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
        #                tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
        #                break_ties=False, random_state=None)
        # pca = PCA(n_components=2)
        # X_train = pca.fit_transform(x_test)
        # clf.fit(x_test, y_test)
        # plot_decision_regions(x_test, y_test, clf=clf, legend=2)
        # plt.xlabel(X.columns[0], size=14)
        # plt.ylabel(X.columns[1], size=14)
        # plt.title('SVM Decision Region Boundary', size=16)
        # plt.show()


    def Train(ETEC_X_Train, ETEC_Y_Train, *,ETEC_C=1.0, ETEC_kernel='linear', ETEC_degree=3, ETEC_gamma='scale', ETEC_coef0=0.0, ETEC_shrinking=True, ETEC_probability=False, ETEC_tol=0.001,
                                          ETEC_cache_size=200, ETEC_class_weight=None, ETEC_verbose=False, ETEC_max_iter=-1, ETEC_decision_function_shape='ovr',
                                          ETEC_break_ties=False, ETEC_random_state=None):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline

        ETEC_SVM = SVC( C=ETEC_C, kernel=ETEC_kernel, degree=ETEC_degree, gamma=ETEC_gamma, coef0=ETEC_coef0, shrinking=ETEC_shrinking, probability=ETEC_probability, tol=ETEC_tol,
                                          cache_size=ETEC_cache_size, class_weight=ETEC_class_weight, verbose=ETEC_verbose, max_iter=ETEC_max_iter, decision_function_shape=ETEC_decision_function_shape,
                                          break_ties=ETEC_break_ties, random_state=ETEC_random_state)
        ETEC_SVM.fit(ETEC_X_Train, ETEC_Y_Train)


        # prediction
        return ETEC_SVM
    def Prediction(ETEC_SVM, ETEX_X_Test):

        return ETEC_SVM.predict(ETEX_X_Test)
    def Confusion_metrix(ETEC_Y_test, y_pred):

        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df
#        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)


    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("Support Vector Classification", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt



class ETEC_KNN:

    # def Draw( x_test,y_test):
    #     from sklearn.svm import SVC



    def Train(ETEC_X_Train, ETEC_Y_Train, ETEC_n_neighbors=5, *, ETEC_weights='uniform', ETEC_algorithm='auto', ETEC_leaf_size=30,
              ETEC_p=2, ETEC_metric='minkowski', ETEC_metric_params=None, ETEC_n_jobs=None):
        from sklearn.neighbors import KNeighborsClassifier

        Knn = KNeighborsClassifier( n_neighbors=ETEC_n_neighbors, weights=ETEC_weights, algorithm=ETEC_algorithm, leaf_size=ETEC_leaf_size,
                        p=ETEC_p, metric=ETEC_metric, metric_params=ETEC_metric_params, n_jobs=ETEC_n_jobs)
        Knn.fit(ETEC_X_Train, ETEC_Y_Train)


        # prediction
        return Knn
    def Prediction(Knn, ETEX_X_Test):

        return Knn.predict(ETEX_X_Test)
    def Confusion_metrix(ETEC_Y_test, y_pred):

        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df
#        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)


    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("KNN", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt


class ETEC_K_means:
    # def Draw(model,x_train,y_train):
    #     print('h')
        # plt.figure(figsize=(12, 12))
        # X_filtered = x_train
        # y_pred = model
        #
        # # plt.subplot((X_filtered,X_filtered))
        # t=ETEC_K_means.Train(x_train,y_train)
        # # print(t.n_clusters)
        #
        # for j in range(t.n_clusters):
        #
        #     plt.scatter(x_train[y_pred], x_train[y_pred ],  label='Cluster '+str(j))
        #     plt.scatter(t.cluster_centers_[ j], t.cluster_centers_[ j], c='yellow', label='Centroids')
        #
        # plt.title("Unevenly Sized Blobs")
        #
        # plt.show()
        # label=model
        # # filter rows of original data
        #
        # u_labels = np.unique(label)
        #
        # # plotting the results:
        #
        # for i in u_labels:
        #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        # plt.legend()
        # plt.show()
    def Train(ETEC_X_Train, ETEC_Y_Train,ETEC_n_clusters=8, *, ETEC_init='k-means++',
              ETEC_n_init=10, ETEC_max_iter=300, ETEC_tol=0.0001, ETEC_verbose=0, ETEC_random_state=None, ETEC_copy_x=True, ETEC_algorithm='auto'):
        from sklearn.cluster import KMeans
       # global n_clusters

        K_means = KMeans(n_clusters=ETEC_n_clusters,  init=ETEC_init, n_init=ETEC_n_init, max_iter=ETEC_max_iter,
                                   tol=ETEC_tol, verbose=ETEC_verbose, random_state=ETEC_random_state, copy_x=ETEC_copy_x, algorithm=ETEC_algorithm)
        K_means.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return K_means

    def Prediction(K_means, ETEX_X_Test):
        return K_means.predict(ETEX_X_Test)

    def Confusion_metrix(ETEC_Y_test, y_pred):
        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df

    #        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)

    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("K-Means", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
       # plt.show()

        return plt



class ETEC_Gradient_Boosting:
    # def Draw(model,x_train,y_train):
        # plt.figure(figsize=(12, 12))
        # X_filtered = x_train
        # y_pred = model
        #
        # # plt.subplot((X_filtered,X_filtered))
        # t=ETEC_K_means.Train(x_train,y_train)
        # # print(t.n_clusters)
        #
        # for j in range(t.n_clusters):
        #
        #     plt.scatter(x_train[y_pred], x_train[y_pred ],  label='Cluster '+str(j))
        #     plt.scatter(t.cluster_centers_[ j], t.cluster_centers_[ j], c='yellow', label='Centroids')
        #
        # plt.title("Unevenly Sized Blobs")
        #
        # plt.show()
        # label=model
        # # filter rows of original data
        #
        # u_labels = np.unique(label)
        #
        # # plotting the results:
        #
        # for i in u_labels:
        #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        # plt.legend()
        # plt.show()
    def Train(ETEC_X_Train, ETEC_Y_Train,*, ETEC_loss='deviance', ETEC_learning_rate=0.1, ETEC_n_estimators=100, ETEC_subsample=1.0, ETEC_criterion='friedman_mse',
              ETEC_min_samples_split=2, ETEC_min_samples_leaf=1, ETEC_min_weight_fraction_leaf=0.0, ETEC_max_depth=3, ETEC_min_impurity_decrease=0.0,
              ETEC_init=None, ETEC_random_state=None, ETEC_max_features=None, ETEC_verbose=0, ETEC_max_leaf_nodes=None,
              ETEC_warm_start=False, ETEC_validation_fraction=0.1, ETEC_n_iter_no_change=None, ETEC_tol=0.0001, ETEC_ccp_alpha=0.0):
        from sklearn.ensemble import GradientBoostingClassifier

        gb = GradientBoostingClassifier( loss=ETEC_loss, learning_rate=ETEC_learning_rate, n_estimators=ETEC_n_estimators, subsample=ETEC_subsample, criterion=ETEC_criterion, min_samples_split=ETEC_min_samples_split,
                         min_samples_leaf=ETEC_min_samples_leaf, min_weight_fraction_leaf=ETEC_min_weight_fraction_leaf, max_depth=ETEC_max_depth, min_impurity_decrease=ETEC_min_impurity_decrease, init=ETEC_init, random_state=ETEC_random_state,
                         max_features=ETEC_max_features, verbose=ETEC_verbose,
                         max_leaf_nodes=ETEC_max_leaf_nodes, warm_start=ETEC_warm_start, validation_fraction=ETEC_validation_fraction, n_iter_no_change=ETEC_n_iter_no_change, tol=ETEC_tol, ccp_alpha=ETEC_ccp_alpha)
        gb.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return gb

    def Prediction(gb, ETEX_X_Test):
        return gb.predict(ETEX_X_Test)

    def Confusion_metrix(ETEC_Y_test, y_pred):
        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df

    #        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)

    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("Gradiant Boosting", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt


class ETEC_AdaBoosting:
    # def Draw(model,x_train,y_train):
    #     plt.figure(figsize=(12, 12))
    #     X_filtered = x_train
    #     y_pred = model
    #
    #     # plt.subplot((X_filtered,X_filtered))
    #     t=ETEC_K_means.Train(x_train,y_train)
    #     # print(t.n_clusters)
    #
    #     for j in range(t.n_clusters):
    #
    #         plt.scatter(x_train[y_pred], x_train[y_pred ],  label='Cluster '+str(j))
    #         plt.scatter(t.cluster_centers_[ j], t.cluster_centers_[ j], c='yellow', label='Centroids')
    #
    #     plt.title("Unevenly Sized Blobs")
    #
    #     plt.show()
        # label=model
        # # filter rows of original data
        #
        # u_labels = np.unique(label)
        #
        # # plotting the results:
        #
        # for i in u_labels:
        #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        # plt.legend()
        # plt.show()
    def Train(ETEC_X_Train, ETEC_Y_Train,ETEC_Y_base_estimator=None, *, ETEC_Y_n_estimators=50,
              ETEC_Y_learning_rate=1.0, ETEC_Y_algorithm='SAMME.R', ETEC_Y_random_state=None):

        from sklearn.ensemble import AdaBoostClassifier

        adaboost = AdaBoostClassifier( base_estimator=ETEC_Y_base_estimator,  n_estimators=ETEC_Y_n_estimators, learning_rate=ETEC_Y_learning_rate, algorithm=ETEC_Y_algorithm, random_state=ETEC_Y_random_state)
        adaboost.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return adaboost

    def Prediction(adaboost, ETEX_X_Test):
        return adaboost.predict(ETEX_X_Test)

    def Confusion_metrix(ETEC_Y_test, y_pred):
        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df

    #        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)

    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("AdaBoosting", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt


class ETEC_Neural_network:
    # def Draw(model,x_train,y_train):
    #     plt.figure(figsize=(12, 12))
    #     X_filtered = x_train
    #     y_pred = model
    #
    #     # plt.subplot((X_filtered,X_filtered))
    #     t=ETEC_K_means.Train(x_train,y_train)
    #     # print(t.n_clusters)
    #
    #     for j in range(t.n_clusters):
    #
    #         plt.scatter(x_train[y_pred], x_train[y_pred ],  label='Cluster '+str(j))
    #         plt.scatter(t.cluster_centers_[ j], t.cluster_centers_[ j], c='yellow', label='Centroids')
    #
    #     plt.title("Unevenly Sized Blobs")
    #
    #     plt.show()
        # label=model
        # # filter rows of original data
        #
        # u_labels = np.unique(label)
        #
        # # plotting the results:
        #
        # for i in u_labels:
        #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        # plt.legend()
        # plt.show()
    def Train(ETEC_X_Train, ETEC_Y_Train,ETEC_hidden_layer_sizes=(100,), ETEC_activation='relu', *, ETEC_solver='adam', ETEC_alpha=0.0001,
              ETEC_batch_size='auto', ETEC_learning_rate='constant', ETEC_learning_rate_init=0.001, ETEC_power_t=0.5,
              ETEC_max_iter=200, ETEC_shuffle=True, ETEC_random_state=None, ETEC_tol=0.0001, ETEC_verbose=False,
              ETEC_warm_start=False, ETEC_momentum=0.9, ETEC_nesterovs_momentum=True, ETEC_early_stopping=False, ETEC_validation_fraction=0.1,
              ETEC_beta_1=0.9, ETEC_beta_2=0.999, ETEC_epsilon=1e-08, ETEC_n_iter_no_change=10, ETEC_max_fun=15000):

        from sklearn.neural_network import MLPClassifier

        nn = MLPClassifier( hidden_layer_sizes=ETEC_hidden_layer_sizes, activation=ETEC_activation,
                                       solver=ETEC_solver, alpha=ETEC_alpha, batch_size=ETEC_batch_size, learning_rate=ETEC_learning_rate,
                                       learning_rate_init=ETEC_learning_rate_init, power_t=ETEC_power_t, max_iter=ETEC_max_iter, shuffle=ETEC_shuffle, random_state=ETEC_random_state,
                                       tol=ETEC_tol, verbose=ETEC_verbose, warm_start=ETEC_warm_start, momentum=ETEC_momentum, nesterovs_momentum=ETEC_nesterovs_momentum,
                                       early_stopping=ETEC_early_stopping, validation_fraction=ETEC_validation_fraction,
                                       beta_1=ETEC_beta_1, beta_2=ETEC_beta_2, epsilon=ETEC_epsilon, n_iter_no_change=ETEC_n_iter_no_change, max_fun=ETEC_max_fun)
        nn.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return nn

    def Prediction(nn, ETEX_X_Test):
        return nn.predict(ETEX_X_Test)

    def Confusion_metrix(ETEC_Y_test, y_pred):
        return confusion_matrix(ETEC_Y_test, y_pred)

    def Accuracy(ETEC_Y_test, y_pred):
        return accuracy_score(ETEC_Y_test, y_pred) * 100

    def Report(ETEC_Y_test, y_pred):
        rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        return df

    #        return classification_report(ETEC_Y_test, y_pred)

    def texture(model_dt):
        text_representation = tree.export_text(model_dt)
        return print(text_representation)

    def Visualization(ETEC_Y_test, y_pred):
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
                    yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
                    fmt="d", cmap="Blues", annot_kws={"size": 20});
        plt.title("Neural Network", fontsize=25)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        # plt.savefig('DT/decission tree.png')
        # InlineBackend.figure_format = "DT"
        #plt.show()

        return plt


class ETEC_LogisticRegression:

    def Train(ETEC_X_Train, ETEC_Y_Train,ETEC_penalty='l2', *, ETEC_dual=False, ETEC_tol=0.0001, ETEC_C=1.0, ETEC_fit_intercept=True, ETEC_intercept_scaling=1,
              ETEC_class_weight=None, ETEC_random_state=None, ETEC_solver='lbfgs',
              ETEC_max_iter=1000, ETEC_multi_class='auto', ETEC_verbose=0, ETEC_warm_start=False, ETEC_n_jobs=None, ETEC_l1_ratio=None):

        from sklearn.linear_model import LogisticRegression


        lG = LogisticRegression(penalty=ETEC_penalty,  dual=ETEC_dual, tol=ETEC_tol, C=ETEC_C, fit_intercept=ETEC_fit_intercept, intercept_scaling=ETEC_intercept_scaling,
              class_weight=ETEC_class_weight, random_state=ETEC_random_state, solver=ETEC_solver,
              max_iter=ETEC_max_iter, multi_class=ETEC_multi_class, verbose=ETEC_verbose, warm_start=ETEC_warm_start, n_jobs=ETEC_n_jobs, l1_ratio=ETEC_l1_ratio)
        lG.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return lG

    def Prediction(classifier, ETEX_X_Test):
        return classifier.predict(ETEX_X_Test)

    def Mean_square_error(ETEC_Y_test, y_pred):
        return mean_squared_error(ETEC_Y_test, y_pred)


    def Accuracy(Y_test, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(Y_test, y_pred)*100

    def coef(classifier):
        return classifier.coef_


class ETEC_LinearRegression:

    def Train(ETEC_X_Train, ETEC_Y_Train, *, ETEC_fit_intercept=True, ETEC_normalize='deprecated', ETEC_copy_X=True,
              ETEC_n_jobs=None, ETEC_positive=False):
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression(fit_intercept=ETEC_fit_intercept, normalize=ETEC_normalize, copy_X=ETEC_copy_X,
                              n_jobs=ETEC_n_jobs, positive=ETEC_positive)
        lr.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return lr

    def Prediction(classifier, ETEX_X_Test):
        return classifier.predict(ETEX_X_Test)

    def Mean_square_error(ETEC_Y_test, y_pred):
        return mean_squared_error(ETEC_Y_test, y_pred)


    def Accuracy(Y_test, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(Y_test, y_pred)*100

    def coef(classifier):
        return classifier.coef_


class ETEC_PoissonRegressor:

    def Train(ETEC_X_Train, ETEC_Y_Train, *, ETEC_alpha=1.0, ETEC_fit_intercept=True, ETEC_max_iter=1000, ETEC_tol=0.0001, ETEC_warm_start=False, ETEC_verbose=0):
        from sklearn.linear_model import PoissonRegressor

        pr = PoissonRegressor( alpha=ETEC_alpha, fit_intercept=ETEC_fit_intercept, max_iter=ETEC_max_iter, tol=ETEC_tol
                               , warm_start=ETEC_warm_start, verbose=ETEC_verbose)
        pr.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return pr

    def Prediction(classifier, ETEX_X_Test):
        return classifier.predict(ETEX_X_Test)

    def Mean_square_error(ETEC_Y_test, y_pred):
        return mean_squared_error(ETEC_Y_test, y_pred)


    def Accuracy(Y_test, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(Y_test, y_pred)*100

    def coef(classifier):
        return classifier.coef_



class ETEC_LassoRegressor:

    def Train(ETEC_X_Train, ETEC_Y_Train,ETEC_alpha=1.0, *, ETEC_fit_intercept=True, ETEC_normalize='deprecated',
              ETEC_precompute=False, ETEC_copy_X=True, ETEC_max_iter=1000, ETEC_tol=0.0001,
              ETEC_warm_start=False, ETEC_positive=False, ETEC_random_state=None, ETEC_selection='cyclic'):
        from sklearn.linear_model import Lasso
        Lasso



        Lr = Lasso( alpha=ETEC_alpha, fit_intercept=ETEC_fit_intercept, normalize=ETEC_normalize,
              precompute=ETEC_precompute, copy_X=ETEC_copy_X, max_iter=ETEC_max_iter,tol=ETEC_tol,
              warm_start=ETEC_warm_start, positive=ETEC_positive, random_state=ETEC_random_state, selection=ETEC_selection)
        Lr.fit(ETEC_X_Train, ETEC_Y_Train)

        # prediction
        return Lr

    def Prediction(classifier, ETEX_X_Test):
        return classifier.predict(ETEX_X_Test)

    # def score(classifier,X_test, y_test):
    #     return classifier.score(X_test,y_test)*100

    def Mean_square_error(ETEC_Y_test, y_pred):
        return mean_squared_error(ETEC_Y_test, y_pred)


    def Accuracy(Y_test, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(Y_test, y_pred)*100

    def coef(classifier):
        return classifier.coef_

    # def Report(ETEC_Y_test, y_pred):
    #     rep = classification_report(ETEC_Y_test, y_pred, output_dict=True)
    #     df = pd.DataFrame(rep).transpose()
    #     return df

    #        return classification_report(ETEC_Y_test, y_pred)



    # def Visualization(ETEC_Y_test, y_pred):
    #     plt.figure(figsize=(8, 8))
    #     sns.heatmap(confusion_matrix(ETEC_Y_test.values, y_pred, ), xticklabels=ETEC_Y_test.drop_duplicates(),
    #                 yticklabels=ETEC_Y_test.drop_duplicates(), annot=True,
    #                 fmt="d", cmap="Blues", annot_kws={"size": 20});
    #     plt.title("Linear Regression", fontsize=25)
    #     plt.ylabel('True class')
    #     plt.xlabel('Predicted class')
    #     # plt.savefig('DT/decission tree.png')
    #     # InlineBackend.figure_format = "DT"
    #     # plt.show()

        # return plt

