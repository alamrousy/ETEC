# # # import yfinance as yf
# # import streamlit as st
# # # from sklearn import datasets
# # import pandas as pd
# # from pages import data,model
# # from streamlit_multipage import MultiPage
#
#
# # import streamlit as st
# #
# # st.set_page_config(
# #     page_title="Multipage App",
# #     page_icon="👋",
# # )
# #
# # st.title("Main Page")
# # st.sidebar.success("Select a page above.")
# #
# # if "my_input" not in st.session_state:
# #     st.session_state["my_input"] = ""
# #
# # my_input = st.text_input("Input a text here", st.session_state["my_input"])
# # submit = st.button("Submit",)
# # if submit:
# #     st.session_state["my_input"] = my_input
# # st.write("You have entered: ", st.session_state['my_input'])
#
# import streamlit as st
#
# st.markdown(
#     """
#     # Welcome to Etec Watson Analysis! 👋"
#      ***for Machine Learning and Data Science projects***  👈
#
#     **ETEC’s ML interface is a user-friendly interface developed by Education & Training Evaluation Commission (ETEC). The interface allows users to apply machine learning algorithms and benefit from the output results.**
#
#     **Allows researchers and analysts to test machine-learning algorithms and models without having the required programming background. Each type of analysis used through the interface will include 0% coding.**
#
#    **Target Audience:**
#
#    *Researchers
#
#    *Data Analysts
#
#    *Business Intelligence Developers
#
#    *Anyone who works in the education-related field.
#
#    *Students learning ML and modelling.
#
#
#
#     ##### This work has been done by the department of the data analytics in ETEC
#     """
#  )


# import streamlit as st
# import pandas as pd
# import altair as alt
#
# session_key = 'data_source'
#
# track_data = 'Track My Data'
# if session_key not in st.session_state:
#     data_source = pd.DataFrame({
#         "Person": ["Bill", "Sally", "Bill", "Sally", "Bill", "Sally", "Bill", "Sally", "Bill"],
#         track_data: [15, 10, 30, 13, 8, 70, 17, 83, 70],
#         "Date": ["2022-1-23", "2022-1-30", "2022-1-5", "2022-2-21", "2022-2-1", "2022-2-2", "2022-3-1", "2022-3-3",
#                  "2022-3-6"]
#     })
#     data_source['Date'] = pd.to_datetime(data_source['Date'])
#     st.session_state[session_key] = data_source
#
#
# def save_session():
#     filtered_line_chart = st.session_state[session_key].query(
#         "Date >= @start_date "
#     )
#     filtered_line_chart = st.session_state[session_key]
#
#
# def clear_data():
#     st.session_state[session_key] = pd.DataFrame()
#
#
# with st.sidebar.form("my_form"):
#     input_person = st.selectbox(
#         'Who would you like to enter a ' + track_data + ' for?',
#         ('Bill', 'Sally'))
#
#     input_weight = st.text_input(track_data + " input")
#     input_date = pd.to_datetime(st.date_input("Date input"))
#
#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         data_source = st.session_state[session_key]
#
#         data_source = data_source.append(
#             {track_data: int(input_weight), "Person": input_person, "Date": pd.to_datetime(input_date)},
#             ignore_index=True)
#
#         st.session_state[session_key] = data_source
#
#     uploaded_file = st.file_uploader("Choose a file")
#     if uploaded_file is not None:
#         # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         st.write(bytes_data)
#
#         dataframe = pd.read_csv(uploaded_file)
#         st.write(dataframe)
#
#
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')
#
#
# csv = convert_df(st.session_state[session_key])
#
# col1, col2, col3 = st.columns(3)
# with col1:
#     start_date = st.date_input(
#         "Show " + track_data + " after this date",
#         None)
#
#     start_date = pd.to_datetime(start_date)
#     filter_date_button = st.button('Filter', on_click=save_session)
#
# with col2:
#     st.write("")
#
# with col3:
#     if st.download_button(
#             label="Download data as CSV",
#             data=csv,
#             file_name=track_data + "_records.csv",
#             mime='text/csv',
#     ):
#         st.write(' Data Downloaded')
#     clear_data = st.button("Clear Data", on_click=clear_data)
#
# if filter_date_button:
#     filtered_line_chart = st.session_state[session_key].query(
#         "Date >= @start_date ")
# else:
#     filtered_line_chart = st.session_state[session_key]
#
# if not st.session_state[session_key].empty:
#     line_chart = alt.Chart(filtered_line_chart).mark_line().encode(
#         y=alt.Y(track_data, title=track_data),
#         x=alt.X('Date', title='Month'),
#         color='Person'
#     ).properties(
#         height=400, width=700,
#         title=track_data + " Chart"
#     ).configure_title(
#         fontSize=16
#     )
#     st.altair_chart(line_chart, use_container_width=True)
# else:
#     st.write("All Data has been cleared")
#
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import streamlit as st
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import seaborn as sns
import ETEC_Watson_Analysis5 as ETEC
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import streamlit.components.v1 as components
import mpld3


st.set_page_config(layout="wide")

#######   Decission Tree   #######

if 'Train_Test_Split_dt' not in st.session_state:
    st.session_state['Train_Test_Split_dt'] = 20
if 'min_sample_split_dt' not in st.session_state:
    st.session_state['min_sample_split_dt'] = 2
if 'min_sample_leaf_dt' not in st.session_state:
    st.session_state['min_sample_leaf_dt'] = 2
if 'max_leaf_node_dt' not in st.session_state:
    st.session_state['max_leaf_node_dt'] = 2
if 'random_state_dt' not in st.session_state:
    st.session_state['random_state_dt'] = 0
#######   Random_Forest   #######

if 'min_sample_split_rf' not in st.session_state:
    st.session_state['min_sample_split_rf'] = 2
if 'min_sample_leaf_rf' not in st.session_state:
    st.session_state['min_sample_leaf_rf'] = 2
if 'max_leaf_node_rf' not in st.session_state:
    st.session_state['max_leaf_node_rf'] = 2
if 'random_state_rf' not in st.session_state:
    st.session_state['random_state_rf'] = 0
#######   Support Vector Machine   #######

if 'Degree_SVM' not in st.session_state:
    st.session_state['Degree_SVM'] = 3
if 'kernal_SVM' not in st.session_state:
    st.session_state['kernal_SVM'] = 'rbf'
if 'gamma_SVM' not in st.session_state:
    st.session_state['gamma_SVM'] = 'scale'
if 'SVM_random_state' not in st.session_state:
    st.session_state['SVM_random_state'] = 0

# KNN_Leaf_size, algorithm_KNN, Weight_KNN, KNN_neighbors
#######   KNN   #######

if 'KNN_Leaf_size' not in st.session_state:
    st.session_state['KNN_Leaf_size'] = 30
if 'algorithm_KNN' not in st.session_state:
    st.session_state['algorithm_KNN'] = 'auto'
if 'Weight_KNN' not in st.session_state:
    st.session_state['Weight_KNN'] = 'uniform'
if 'neighbors_KNN' not in st.session_state:
    st.session_state['neighbors_KNN'] = 5
# st.set_page_config(layout="wide")

#######   GB   #######

if 'criterion_GB' not in st.session_state:
    st.session_state['criterion_GB'] = 'friedman_mse'
if 'GB_learning_rate' not in st.session_state:
    st.session_state['GB_learning_rate'] = 0.1
if 'loss_GB' not in st.session_state:
    st.session_state['loss_GB'] = 'exponential'
if 'n_estimators_GB' not in st.session_state:
    st.session_state['n_estimators_GB'] = 100

#######   AB   #######

# if 'criterion_AB' not in st.session_state:
#     st.session_state['criterion_AB'] = 'friedman_mse'
if 'AB_learning_rate' not in st.session_state:
    st.session_state['AB_learning_rate'] = 1.0
if 'algorithm_AB' not in st.session_state:
    st.session_state['algorithm_AB'] = 'SAMME.R'
if 'n_estimators_AB' not in st.session_state:
    st.session_state['n_estimators_AB'] = 50

#######   NN   #######

if 'NN_hidden_layer_sizes' not in st.session_state:
    st.session_state['NN_hidden_layer_sizes'] = 100
if 'activation_NN' not in st.session_state:
    st.session_state['activation_NN'] = 'relu'
if 'solver_NN' not in st.session_state:
    st.session_state['solver_NN'] = 'adam'
if 'max_iter_NN' not in st.session_state:
    st.session_state['max_iter_NN'] = 200

###########   LR    ##############
if 'Train_Test_Split_LR' not in st.session_state:
    st.session_state['Train_Test_Split_LR'] = 20
if 'fit_intercept_LR' not in st.session_state:
    st.session_state['fit_intercept_LR'] = 'True'

#######   LOGR   #######

if 'solver_LOGR' not in st.session_state:
    st.session_state['solver_LOGR'] = 'lbfgs'
if 'penalty_LOGR' not in st.session_state:
    st.session_state['penalty_LOGR'] = 'l2'
if 'multi_class_LOGR' not in st.session_state:
    st.session_state['multi_class_LOGR'] = 'auto'
if 'max_iter_LOGR' not in st.session_state:
    st.session_state['max_iter_LOGR'] = 100


#######   LASSO   #######

if 'max_iter_LASSOR' not in st.session_state:
    st.session_state['max_iter_LASSOR'] = 1000
# if 'precompute_LASSOR' not in st.session_state:
#     st.session_state['precompute_LASSOR'] = 'False'
if 'selection_LASSOR' not in st.session_state:
    st.session_state['selection_LASSOR'] = 'cyclic'
if 'fit_intercept_LASSOR' not in st.session_state:
    st.session_state['fit_intercept_LASSOR'] = 'True'

########## POISONR regression #################
if 'max_iter_POISONR' not in st.session_state:
    st.session_state['max_iter_POISONR'] = 100
if 'fit_intercept_POISONR' not in st.session_state:
    st.session_state['fit_intercept_POISONR'] = 'True'


#######   ElasticNetCV   #######

# if 'precompute_ElasticNetCV' not in st.session_state:
#     st.session_state['precompute_ElasticNetCV'] = 'auto'
if 'max_iter_ElasticNetCV' not in st.session_state:
    st.session_state['max_iter_ElasticNetCV'] = 1000
if 'selection_ElasticNetCV' not in st.session_state:
    st.session_state['selection_ElasticNetCV'] = 'cyclic'
if 'fit_intercept_ElasticNetCV' not in st.session_state:
    st.session_state['fit_intercept_ElasticNetCV'] = 'true'
#######   BayesianRidge   #######

if 'max_iter_BayesianRidge' not in st.session_state:
    st.session_state['max_iter_BayesianRidge'] = 300
if 'fit_intercept_BayesianRidge' not in st.session_state:
    st.session_state['fit_intercept_BayesianRidge'] = 'true'
#######   GaussianProcess   #######

# normalize_y_GaussianProcess, random_state_GaussianProcess
if 'normalize_y_GaussianProcess' not in st.session_state:
    st.session_state['normalize_y_GaussianProcess'] = 'False'
if 'random_state_GaussianProcess' not in st.session_state:
    st.session_state['random_state_GaussianProcess'] = 0

# min_samples_split_Random_Forest_R, n_estimators_Random_Forest_R, max_features_Random_Forest_R, Random_Forest_R_criterion
#######   Random_Forest_R   #######

if 'min_samples_split_Random_Forest_R' not in st.session_state:
    st.session_state['min_samples_split_Random_Forest_R'] = 2
if 'n_estimators_Random_Forest_R' not in st.session_state:
    st.session_state['n_estimators_Random_Forest_R'] = 100
if 'max_features_Random_Forest_R' not in st.session_state:
    st.session_state['max_features_Random_Forest_R'] = 1.0
if 'criterion_Random_Forest_R' not in st.session_state:
    st.session_state['criterion_Random_Forest_R'] = 'squared_error'

# alpha_Ridge_R
#######   Ridge_R   #######

if 'alpha_Ridge_R' not in st.session_state:
    st.session_state['alpha_Ridge_R'] = 0.5

# hidden_layer_sizes_Neural_R, solver_Neural_R, activation_Neural_R, learning_rate_Neural_R, max_iter_Neural_R

#######   Neural_R   #######

if 'hidden_layer_sizes_Neural_R' not in st.session_state:
    st.session_state['hidden_layer_sizes_Neural_R'] = 100
if 'solver_Neural_R' not in st.session_state:
    st.session_state['solver_Neural_R'] = 'adam'
if 'activation_Neural_R' not in st.session_state:
    st.session_state['activation_Neural_R'] = 'relu'
if 'learning_rate_Neural_R' not in st.session_state:
    st.session_state['learning_rate_Neural_R'] = 'constant'
if 'max_iter_Neural_R' not in st.session_state:
    st.session_state['max_iter_Neural_R'] = 200

# degree_SVR_R,gamma_SVR_R
#######   SVR_R   #######

if 'gamma_SVR_R' not in st.session_state:
    st.session_state['gamma_SVR_R'] = 'scale'
if 'degree_SVR_R' not in st.session_state:
    st.session_state['degree_SVR_R'] = 3
if 'kernel_SVR_R' not in st.session_state:
    st.session_state['kernel_SVR_R'] = 'rbf'

# n_neighbors_KNN_R, weights_KNN_R, algorithm_KNN_R
#######   KNN_R   #######

if 'n_neighbors_KNN_R' not in st.session_state:
    st.session_state['n_neighbors_KNN_R'] = 5
if 'weights_KNN_R' not in st.session_state:
    st.session_state['weights_KNN_R'] = 'uniform'
if 'algorithm_KNN_R' not in st.session_state:
    st.session_state['algorithm_KNN_R'] = 'auto'

# Functions Exploratory Analysis
class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=[np.object]).columns


    def box_plot(self, main_var, col_x=None, hue=None):
        return px.box(self.df, x=col_x, y=main_var, color=hue)

    # @st.cache
    def violin(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.violinplot(x=col_x, y=main_var, hue=hue,
                              data=self.df, palette="husl", split=split)

    # @st.cache
    def swarmplot(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.swarmplot(x=col_x, y=main_var, hue=hue,
                             data=self.df, palette="husl", dodge=split)

    # @st.cache
    def histogram_num(self, main_var, hue=None, bins=None, ranger=None):
        return px.histogram(self.df[self.df[main_var].between(left=ranger[0], right=ranger[1])], \
                            x=main_var, nbins=bins, color=hue, marginal='violin')

    # @st.cache
    def scatter_plot(self, col_x, col_y, hue=None, size=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue, size=size)

    # @st.cache
    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y, color=hue)

    # @st.cache
    def line_plot(self, col_y, col_x, hue=None, group=None):
        return px.line(self.df, x=col_x, y=col_y, color=hue, line_group=group)

    # @st.cache
    def CountPlot(self, main_var, hue=None):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.countplot(x=main_var, data=self.df, hue=hue, palette='pastel')
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    # @st.cache
    def heatmap_vars(self, cols, func=np.mean):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(
            self.df.pivot_table(index=cols[0], columns=cols[1], values=cols[2], aggfunc=func, fill_value=0).dropna(
                axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    # @st.cache
    def Corr(self, cols=None, method='pearson'):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        if len(cols) != 0:
            corr = self.df[cols].corr(method=method)
        else:
            corr = self.df.corr(method=method)
        chart = sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=30)
        return chart

    # @st.cache
    def DistPlot(self, main_var):
        sns.set(style="whitegrid")
        return sns.distplot(self.df[main_var], color='c', rug=True)

# @st.cache(persist=True)
def get_data(file):

    read_cache_csv = st.cache(pd.read_csv, allow_output_mutation=True)
    df = read_cache_csv(file)
    return df


# @st.cache
def get_stats(df):
    stats_num = df.describe()
    if df.select_dtypes(np.object).empty:
        return stats_num.transpose(), None
    if df.select_dtypes(np.number).empty:
        return None, df.describe(include=np.object).transpose()
    else:
        return stats_num.transpose(), df.describe(include=np.object).transpose()


# @st.cache
def get_info(df):
    return pd.DataFrame(
        {'types': df.dtypes, 'nan': df.isna().sum(), 'nan%': round((df.isna().sum() / len(df)) * 100, 2),
         'unique': df.nunique()})

# @st.cache
def input_null(df, col, radio):
    df_inp = df.copy()

    if radio == 'Mean':
        st.write("Mean:", df[col].mean())
        df_inp[col] = df[col].fillna(df[col].mean())

    elif radio == 'Median':
        st.write("Median:", df[col].median())
        df_inp[col] = df[col].fillna(df[col].median())

    elif radio == 'Mode':
        for i in col:
            st.write(f"Mode {i}:", df[i].mode()[0])
            df_inp[i] = df[i].fillna(df[i].mode()[0])

    elif radio == 'Repeat last valid value':
        df_inp[col] = df[col].fillna(method='ffill')

    elif radio == 'Repeat next valid value':
        df_inp[col] = df[col].fillna(method='bfill')

    elif radio == 'Value':
        for i in col:
            number = st.number_input(f'Insert a number to fill missing values in {i}', format='%f', key=i)
            df_inp[i] = df[i].fillna(number)

    elif radio == 'Drop rows with missing values':
        if type(col) != list:
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(get_na_info(df_inp, df, col))

    return df_inp

# @st.cache
def input_null_cat(df, col, radio):
    df_inp = df.copy()

    if radio == 'Text':
        for i in col:
            user_text = st.text_input(f'Replace missing values in {i} with', key=i)
            df_inp[i] = df[i].fillna(user_text)

    elif radio == 'Drop rows with missing values':
        if type(col) != list:
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(pd.concat([get_info(df[col]), get_info(df_inp[col])], axis=0))

    return df_inp


# @st.cache
def get_na_info(df_preproc, df, col):
    raw_info = pd_of_stats(df, col)
    prep_info = pd_of_stats(df_preproc, col)
    return raw_info.join(prep_info, lsuffix='_raw', rsuffix='_prep').T


# @st.cache
def pd_of_stats(df, col):
    # Descriptive Statistics
    stats = dict()
    stats['Mean'] = df[col].mean()
    stats['Std'] = df[col].std()
    stats['Var'] = df[col].var()
    stats['Kurtosis'] = df[col].kurtosis()
    stats['Skewness'] = df[col].skew()
    stats['Coefficient Variance'] = stats['Std'] / stats['Mean']
    return pd.DataFrame(stats, index=col).T.round(2)


# @st.cache
def pf_of_info(df, col):
    info = dict()
    # df=pd.DataFrame(df)
    # df.astype(str)
    # info['Type'] = df[df[col]].dtypes
    info['Unique'] = df[col].nunique()
    info['n_zeros'] = (len(df) - np.count_nonzero(df[col]))
    info['p_zeros'] = round(info['n_zeros'] * 100 / len(df), 2)
    info['nan'] = df[col].isna().sum()
    info['p_nan'] = (df[col].isna().sum() / df.shape[0]) * 100
    return pd.DataFrame(info, index=col).T.round(2)


# @st.cache
def pd_of_stats_quantile(df, col):
    df_no_na = df[col].dropna()
    stats_q = dict()

    stats_q['Min'] = df[col].min()
    label = {0.25: "Q1", 0.5: 'Median', 0.75: "Q3"}
    for percentile in np.array([0.25, 0.5, 0.75]):
        stats_q[label[percentile]] = df_no_na.quantile(percentile)
    stats_q['Max'] = df[col].max()
    stats_q['Range'] = stats_q['Max'] - stats_q['Min']
    stats_q['IQR'] = stats_q['Q3'] - stats_q['Q1']
    return pd.DataFrame(stats_q, index=col).T.round(2)


# @st.cache
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

# @st.cache
def plot_univariate(obj_plot, main_var, radio_plot_uni):
    if radio_plot_uni == 'Histogram':
        st.subheader('Histogram')
        bins, range_ = None, None
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None))
        bins_ = st.sidebar.slider('Number of bins optional', value=50)
        range_ = st.sidebar.slider('Choose range optional', float(obj_plot.df[main_var].min()), \
                                   float(obj_plot.df[main_var].max()),
                                   (float(obj_plot.df[main_var].min()), float(obj_plot.df[main_var].max())))
        if st.sidebar.button('Plot histogram chart'):
            st.plotly_chart(obj_plot.histogram_num(main_var, hue_opt, bins_, range_))

    if radio_plot_uni == ('Distribution Plot'):
        st.subheader('Distribution Plot')
        if st.sidebar.button('Plot distribution'):
            fig = obj_plot.DistPlot(main_var)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    if radio_plot_uni == 'BoxPlot':
        st.subheader('Boxplot')
        # col_x, hue_opt = None, None
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0, None),
                                     key='boxplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='boxplot1')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(main_var, col_x, hue_opt))

# @st.cache(suppress_st_warning=True)
def plot_multivariate(obj_plot, radio_plot):
    if radio_plot == ('Boxplot'):
        st.subheader('Boxplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)", obj_plot.num_vars, key='boxplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0, None),
                                     key='boxplot2')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='boxplot1')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(col_y, col_x, hue_opt))

    if radio_plot == ('Violin'):
        st.subheader('Violin')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)", obj_plot.num_vars, key='violin')
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0, None),
                                     key='violin1')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='violin2')
        split = st.sidebar.checkbox("Split", key='violin3')
        if st.sidebar.button('Plot violin chart'):
            fig = obj_plot.violin(col_y, col_x, hue_opt, split)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    if radio_plot == ('Swarmplot'):
        st.subheader('Swarmplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)", obj_plot.num_vars )
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0, None),
                                     key='swarmplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None) )
        split = st.sidebar.checkbox("Split")
        if st.sidebar.button('Plot swarmplot chart'):
            fig = obj_plot.swarmplot(col_y, col_x, hue_opt, split)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    def pretty(method):
        return method.capitalize()

    if radio_plot == ('Correlation'):
        st.subheader('Heatmap Correlation Plot')
        correlation = st.sidebar.selectbox("Choose the correlation method", ('pearson', 'kendall', 'spearman'),
                                           format_func=pretty)
        cols_list = st.sidebar.multiselect("Select columns", obj_plot.columns)
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        if st.sidebar.button('Plot heatmap chart'):
            fig = obj_plot.Corr(cols_list, correlation)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    def map_func(function):
        dic = {np.mean: 'Mean', np.sum: 'Sum', np.median: 'Median'}
        return dic[function]

    if radio_plot == ('Heatmap'):
        st.subheader('Heatmap between vars')
        st.markdown(" In order to plot this chart remember that the order of the selection matters, \
            chooose in order the variables that will build the pivot table: row, column and value.")
        cols_list = st.sidebar.multiselect("Select 3 variables (2 categorical and 1 numeric)", obj_plot.columns,
                                           key='heatmapvars')
        agg_func = st.sidebar.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median),
                                        format_func=map_func)
        if st.sidebar.button('Plot heatmap between vars'):
            fig = obj_plot.heatmap_vars(cols_list, agg_func)
            st.pyplot()

    if radio_plot == ('Histogram'):
        st.subheader('Histogram')
        col_hist = st.sidebar.selectbox("Choose main variable", obj_plot.num_vars, key='hist')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='hist2')
        bins_, range_ = None, None
        bins_ = st.sidebar.slider('Number of bins optional', value=30)
        range_ = st.sidebar.slider('Choose range optional', int(obj_plot.df[col_hist].min()),
                                   int(obj_plot.df[col_hist].max()), \
                                   (int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max())))
        if st.sidebar.button('Plot histogram chart'):
            st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

    if radio_plot == ('Scatterplot'):
        st.subheader('Scatter plot')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.num_vars, key='scatter')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.num_vars, key='scatter1')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='scatter2')
        size_opt = st.sidebar.selectbox("Size (numerical) optional", obj_plot.columns.insert(0, None), key='scatter3')
        if st.sidebar.button('Plot scatter chart'):
            st.plotly_chart(obj_plot.scatter_plot(col_x, col_y, hue_opt, size_opt))

    if radio_plot == ('Countplot'):
        st.subheader('Count Plot')
        col_count_plot = st.sidebar.selectbox("Choose main variable", obj_plot.columns, key='countplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='countplot1')
        if st.sidebar.button('Plot Countplot'):
            fig = obj_plot.CountPlot(col_count_plot, hue_opt)
            st.pyplot()

    if radio_plot == ('Barplot'):
        st.subheader('Barplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)", obj_plot.num_vars, key='barplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns, key='barplot1')
        hue_opt = st.sidebar.selectbox("Hue (categorical/numerical) optional", obj_plot.columns.insert(0, None),
                                       key='barplot2')
        if st.sidebar.button('Plot barplot chart'):
            st.plotly_chart(obj_plot.bar_plot(col_y, col_x, hue_opt))

    if radio_plot == ('Lineplot'):
        st.subheader('Lineplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)", obj_plot.num_vars, key='lineplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns, key='lineplot1')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0, None), key='lineplot2')
        group = st.sidebar.selectbox("Group color (categorical) optional", obj_plot.columns.insert(0, None),
                                     key='lineplot3')
        if st.sidebar.button('Plot lineplot chart'):
            st.plotly_chart(obj_plot.line_plot(col_y, col_x, hue_opt, group))


# def plot_roc_curve( y_test,y_pred):
#     # y_pred = model.predict_proba(X_test)[:, 1]
#     fpr, tpr, _ = roc_curve(y_test, y_pred)
#     auc = round(roc_auc_score(y_test, y_pred), 4)
#     plt.plot(fpr, tpr, label="Gradient Boosting, AUC=" + str(auc))
#
#     # add legend
#     plt.legend()
def main():
    st.title('Exploratory Data Analysis :mag:')
    st.header('Analyze the descriptive statistics and the distribution of your data. Preview and save your graphics.')

    file = st.file_uploader('Upload your file (.csv)', type='csv')




    if file is not None:
        df = get_data(file)

        expander=st.expander('data')
        expander.write(df)

        tab1, tab2, tab3 = st.tabs(["Data Summary", "model", "Owl"])
        with tab1:

            numeric_features = df.select_dtypes(include=[np.number]).columns
            categorical_features = df.select_dtypes(include=[np.object]).columns

            def basic_info(df):
                st.header("Data")
                st.write('Number of observations', df.shape[0])
                st.write('Number of variables', df.shape[1])
                st.write('Number of missing (%)', ((df.isna().sum().sum() / df.size) * 100).round(2))

            # Visualize data
            basic_info(df)

            # Sidebar Menu
            options = ["View statistics", "Statistic univariate", "Statistic multivariate"]
            menu = tab1.selectbox("Menu options", options)


            # Data statistics

            df_info = get_info(df)
            if (menu == "View statistics"):
                df_stat_num, df_stat_obj = get_stats(df)
                st.markdown('**Numerical summary**')
                st.table(df_stat_num)
                st.markdown('**Categorical summary**')
                st.table(df_stat_obj)
                st.markdown('**Missing Values**')
                r=df_info.astype(str)
                st.table(r)

            eda_plot = EDA(df)

            # Visualize data

            if (menu == "Statistic univariate"):
                st.header("Statistic univariate")
                st.markdown("Provides summary statistics of only one variable in the raw dataset.")
                main_var = st.selectbox("Choose one variable to analyze:", df.columns.insert(0, None))

                if main_var in numeric_features:
                    if main_var != None:
                        st.subheader("Variable info")
                        # tyt=
                        st.table(pf_of_info(df, [main_var]))
                        st.subheader("Descriptive Statistics")
                        st.table((pd_of_stats(df, [main_var])))
                        st.subheader("Quantile Statistics")
                        st.table((pd_of_stats_quantile(df, [main_var])))

                        chart_univariate = st.sidebar.radio('Chart', ('None', 'Histogram', 'BoxPlot', 'Distribution Plot'))

                        plot_univariate(eda_plot, main_var, chart_univariate)

                if main_var in categorical_features:
                    st.table(df[main_var].describe(include='all').fillna("").astype("str"))
                    st.bar_chart(df[main_var].value_counts().to_frame())

                tab1.subheader("Explore other categorical variables!")
                var = tab1.selectbox("Check its unique values and its frequency:", df.columns.insert(0, None))
                if var != None:
                    aux_chart = df[var].value_counts(dropna=False).to_frame()
                    data = tab1.table(aux_chart.style.bar(color='#3d66af'))

            if (menu == "Statistic multivariate"):
                st.header("Statistic multivariate")

                st.markdown(
                    'Here you can visualize your data by choosing one of the chart options available on the sidebar!')

                st.sidebar.subheader('Data visualization options')
                radio_plot = st.sidebar.radio('Choose plot style',
                                              ('Correlation', 'Boxplot', 'Violin', 'Swarmplot', 'Heatmap', 'Histogram', \
                                                 'Scatterplot', 'Countplot', 'Barplot', 'Lineplot'))

                plot_multivariate(eda_plot, radio_plot)

            # st.sidebar.title('Hi, everyone!')
            # st.sidebar.info('I hope this app data explorer tool is userful for you! \n \
            #     You find me here: \n \
            #     www.linkedin.com/in/marinaramalhete')

        with tab2:
          classification_tab, regression_tab = st.tabs(['Classification', 'Regression'])
          with classification_tab:
              if file is not None:
                  # st.sidebar.write('hello mother fucker')
                  features = st.multiselect(
                      'features', df.columns, key='DT_Feature'
                  )
                  coltarget, coltrain_test_split = st.columns(2)
                  with coltarget:
                    target = st.selectbox(
                      'Target Value',
                      df.columns, key='DT_Target')


                  with coltrain_test_split:

                    Train_Test_Split_dt = st.slider('Train Test Split', 0, 100, st.session_state['Train_Test_Split_dt'],
                                                  key='DT_Train_Test_Split')
                    st.session_state['Train_Test_Split_dt'] = Train_Test_Split_dt
                    Data_model = pd.concat([df[features], df[target]], axis=1)
                  if len(features)>0:
                      DT,RF, SVM,KNN,GB,AB,NN ,Logestic_R= st.tabs(["Decission Tree", "Random Forest", "Support Vector Machine",'K-nearest Neighbour','Gradiant Boosting','Adaptive Boosting','Neural Network','Logestic Regression'])

                      with DT:

                            colDT1,colDT2=st.columns(2)

                            col_DT1,col_DT2,col_DT3=st.columns((1,1,2))
                            with col_DT1:
                                min_sample_split_dt = st.slider('min sample split', 0, 100, st.session_state['min_sample_split_dt'],
                                                                key='DT_min_sample_split')
                                st.session_state['min_sample_split_dt'] = min_sample_split_dt

                                min_sample_leaf_dt = st.slider('min sample leaf', 0, 100, st.session_state['min_sample_leaf_dt'])
                                st.session_state['min_sample_leaf_dt'] = min_sample_leaf_dt
                                max_leaf_node_dt = st.slider('max leaf node', 2, 100, st.session_state['max_leaf_node_dt'])
                                st.session_state['max_leaf_node_dt'] = max_leaf_node_dt
                            with col_DT2:
                                random_state = st.number_input('random state', value=st.session_state['random_state_dt'])
                                st.session_state['random_state_dt'] = random_state
                                option = st.selectbox(
                                    'criteria',
                                    ('gini', 'entropy', 'log_loss'))

                                # st.button('Build Model',key='DT_BUTTON')

                            if st.button('Build Model',key='DT_BUILD'):
                                x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                         Train_Test_Split_dt / 100)
                                DT_Model = ETEC.ETEC_Decission_Tree.Train(x_train, y_train,
                                                                             ETEC_min_samples_split=min_sample_split_dt,
                                                                              ETEC_min_samples_leaf=min_sample_leaf_dt,
                                                                              ETEC_max_leaf_nodes=max_leaf_node_dt,
                                                                              ETEC_random_state=random_state)
                                prediction_dt = ETEC.ETEC_Decission_Tree.Prediction(DT_Model, x_test)
                                accuracy_dt = ETEC.ETEC_Decission_Tree.Accuracy(y_test, prediction_dt)
                                Confusion_metrix_dt = ETEC.ETEC_Decission_Tree.Accuracy(y_test, prediction_dt)
                                report_dt=ETEC.ETEC_Decission_Tree.Report(y_test, prediction_dt)
                                visualization =ETEC.ETEC_Decission_Tree.Visualization(y_test, prediction_dt)
                                predicted_value_dt_tab,Accuracy_dt_tab=st.columns([1,1])

                                with predicted_value_dt_tab:
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=accuracy_dt,
                                        title={'text': "Accuracy"},
                                        domain={'x': [0, .6], 'y': [.1, 0.9]}
                                    ))
                                    st.plotly_chart(fig)

                                with Accuracy_dt_tab:

                                    # st.write(accuracy_dt)

                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.markdown('Confusion Metrix')
                                    st.pyplot(visualization,clear_figure=True)
                                    # st.write(confusion_matrix(y_test, prediction_dt))
                                    # st.write(Confusion_metrix_dt)
                                vis_acc,confusion_metrix_dt_tab=st.columns([0.5,0.9])
                                with vis_acc:
                                    # plot_confusion_matrix(Data_model,y_test,prediction_dt)
                                    # st.pyplot()
                                    # st.write(accuracy_dt)
                                    y = y_test.to_numpy()
                                    coll_pred_actual = np.vstack([y, prediction_dt])
                                    coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual), columns=['actual', 'predected'])
                                    st.markdown('Predicted Value')

                                    st.dataframe(coll_pred_actual)

                                with confusion_metrix_dt_tab:
                                    # st.metric(label="Accuracy", value=accuracy_dt)
                                        st.markdown('Report Summary')

                                        st.dataframe(report_dt)

                                        st.subheader("ROC Curve")
                                roc1, pre_reca=st.columns(2)
                                with roc1:
                                        roc_cu_DT = ETEC.ETEC_Decission_Tree.roc_cur(x_train, y_train, x_test, y_test)

                                        st.write('roc curve')
                                        st.pyplot(roc_cu_DT,clear_figure=True)
                                with  pre_reca:
                                        pre_recall_DT = ETEC.ETEC_Decission_Tree.preci_recall(x_train, y_train, x_test, y_test)

                                        st.write('precision_recall')
                                        st.pyplot(pre_recall_DT, clear_figure=True)
                                pred_error1,pred_error2=st.columns(2)
                                with pred_error1:
                                        pred_err_DT = ETEC.ETEC_Decission_Tree.predect_error(x_train, y_train, x_test, y_test)

                                        st.write('precision_recall')
                                        st.pyplot(pred_err_DT, clear_figure=True)

                                        # st.write(DT_Model)
                                        # fig= plot_roc_curve(DT_Model, x_test, y_test)

                                        # st.pyplot(fig)

                                        # st.subheader("Precision-Recall Curve")
                                        # plot_roc_curve(y_test,prediction_dt)
                                        # st.pyplot()

                            with col_DT3:
                                view=pd.concat([df[features], df[target]], axis=1)
                                st.dataframe(view)
                                # st.write(df[target].nunique())
                      with RF:

                            col_rf1, col_rf2, col_rf3 = st.columns((1, 1, 2))
                            with col_rf1:
                                        min_sample_split_rf = st.slider('min sample split', 0, 100, st.session_state['min_sample_split_rf'],
                                                                key='RF_min_sample_split')
                                        st.session_state['min_sample_split_rf'] = min_sample_split_rf
                                        min_sample_leaf_rf = st.slider('min sample leaf', 0, 100, st.session_state['min_sample_leaf_rf'],key='RF_min_sample_leaf')
                                        st.session_state['min_sample_leaf_rf'] = min_sample_leaf_rf
                                        max_leaf_node_rf = st.slider('max leaf node', 0, 100, st.session_state['max_leaf_node_rf'],key='RF_max_leaf_node')
                                        st.session_state['max_leaf_node_rf'] = max_leaf_node_rf

                            with col_rf3:
                                Data_model = pd.concat([df[features], df[target]], axis=1)
                                st.write(Data_model)
                            with col_rf2:

                                random_state_rf = st.number_input('random state', value=st.session_state['random_state_rf'],key='random_state_rf')
                                # st.session_state['random_state_rf'] = random_state_rf
                                option = st.selectbox(
                                    'criteria',
                                    ('gini', 'entropy', 'log_loss'),key='RF_criteria')

                            # if st.button('Build Model',key='RF_BUILD'):
                                    # x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                    #                                                          Train_Test_Split_RF / 100)
                            if st.button('Build Model', key='RF_BUILD'):
                                    x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                         Train_Test_Split_dt / 100)
                                    RF_Model = ETEC.ETEC_Random_Forest.Train(x_train, y_train,
                                        ETEC_min_samples_split=min_sample_split_rf,
                                        ETEC_min_samples_leaf=min_sample_leaf_rf,
                                        ETEC_max_leaf_nodes=max_leaf_node_rf,
                                    ETEC_random_state=random_state)
                                    prediction_RF = ETEC.ETEC_Random_Forest.Prediction(RF_Model, x_test)
                                    accuracy_RF = ETEC.ETEC_Random_Forest.Accuracy(y_test, prediction_RF)
                                    Confusion_metrix_RF = ETEC.ETEC_Random_Forest.Accuracy(y_test, prediction_RF)
                                    report_RF = ETEC.ETEC_Random_Forest.Report(y_test, prediction_RF)
                                    visualization = ETEC.ETEC_Random_Forest.Visualization(y_test, prediction_RF)
                                    predicted_value_RF_tab, Accuracy_RF_tab = st.columns([1, 1])

                                    with predicted_value_RF_tab:
                                            fig = go.Figure(go.Indicator(
                                                mode="gauge+number",
                                                value=accuracy_RF,
                                                title={'text': "Accuracy"},
                                                domain={'x': [0, .6], 'y': [.1, 0.9]}
                                            ))
                                            st.plotly_chart(fig)

                                    with Accuracy_RF_tab:

                                            st.set_option('deprecation.showPyplotGlobalUse', False)
                                            st.markdown('Confusion Metrix')
                                            st.pyplot(visualization, clear_figure=True)

                                    vis_acc, confusion_metrix_RF_tab = st.columns([0.5, 0.9])
                                    with vis_acc:

                                            y = y_test.to_numpy()
                                            coll_pred_actual = np.vstack([y, prediction_RF])
                                            coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                            columns=['actual', 'predected'])
                                            st.markdown('Predicted Value')

                                            st.dataframe(coll_pred_actual)

                                    with confusion_metrix_RF_tab:
                                            # st.metric(label="Accuracy", value=accuracy_RF)
                                            st.markdown('Report Summary')

                                            st.dataframe(report_RF)

                                            st.subheader("ROC Curve")
                                    roc1, pre_reca = st.columns(2)
                                    with roc1:
                                        roc_cu_RF = ETEC.ETEC_Random_Forest.roc_cur(x_train, y_train, x_test, y_test)

                                        st.write('roc curve')
                                        st.pyplot(roc_cu_RF, clear_figure=True)
                                    with  pre_reca:
                                        pre_recall_RF = ETEC.ETEC_Random_Forest.preci_recall(x_train, y_train, x_test, y_test)

                                        st.write('precision_recall')
                                        st.pyplot(pre_recall_RF, clear_figure=True)
                                    pred_error1, pred_error2 = st.columns(2)
                                    with pred_error1:
                                        pred_err_RF = ETEC.ETEC_Random_Forest.predect_error(x_train, y_train, x_test, y_test)

                                        st.write('precision_recall')
                                        st.pyplot(pred_err_RF, clear_figure=True)

                                        # st.write(RF_Model)
                                        # fig= plot_roc_curve(RF_Model, x_test, y_test)

                                        # st.pyplot(fig)

                                        # st.subheader("Precision-Recall Curve")
                                        # plot_roc_curve(y_test,prediction_RF)
                                        # st.pyplot()

                      with SVM:

                            col_SVM1, col_SVM2, col_SVM3 = st.columns((1, 1, 2))
                            with col_SVM1:
                                kernal_SVM = st.selectbox('gamma', ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
                                                          key='SVM_gamma')
                                st.session_state['kernal_SVM'] = kernal_SVM
                                gamma_SVM = st.selectbox('Select the Kernel coefficient for rbf, poly and sigmoid',['auto','scale'] ,key='SVM_kernal')

                                st.session_state['gamma_SVM'] = gamma_SVM
                            with col_SVM2:
                                SVM_random_state = st.number_input('random state', value=st.session_state['SVM_random_state'],key='random_state_SVM')
                                st.session_state['SVM_random_state'] = SVM_random_state
                                Degree_SVM = st.slider('Degree of the polynomial', 0, 10, st.session_state['Degree_SVM'],
                                                       key='SVM_min_sample_split')
                                st.session_state['Degree_SVM'] = Degree_SVM

                                # st.button('Build Model',key='build_SVM')
                            if st.button('Build Model', key='SVM_BUILD'):
                                    x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                         Train_Test_Split_dt / 100)
                                    SVM_Model = ETEC.ETEC_SVM.Train(x_train, y_train,
                                        ETEC_degree=Degree_SVM,
                                        ETEC_gamma=gamma_SVM,
                                        ETEC_kernel=kernal_SVM,
                                    ETEC_random_state=random_state)
                                    prediction_SVM = ETEC.ETEC_SVM.Prediction(SVM_Model, x_test)
                                    accuracy_SVM = ETEC.ETEC_SVM.Accuracy(y_test, prediction_SVM)
                                    Confusion_metrix_SVM = ETEC.ETEC_SVM.Accuracy(y_test, prediction_SVM)
                                    report_SVM = ETEC.ETEC_SVM.Report(y_test, prediction_SVM)
                                    visualization = ETEC.ETEC_SVM.Visualization(y_test, prediction_SVM)
                                    predicted_value_SVM_tab, Accuracy_SVM_tab = st.columns([1, 1])

                                    with predicted_value_SVM_tab:
                                            fig = go.Figure(go.Indicator(
                                                mode="gauge+number",
                                                value=accuracy_SVM,
                                                title={'text': "Accuracy"},
                                                domain={'x': [0, .6], 'y': [.1, 0.9]}
                                            ))
                                            st.plotly_chart(fig)

                                    with Accuracy_SVM_tab:

                                            st.set_option('deprecation.showPyplotGlobalUse', False)
                                            st.markdown('Confusion Metrix')
                                            st.pyplot(visualization,clear_figure=True)

                                    vis_acc, confusion_metrix_SVM_tab = st.columns([0.5, 0.9])
                                    with vis_acc:

                                            y = y_test.to_numpy()
                                            coll_pred_actual = np.vstack([y, prediction_SVM])
                                            coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                            columns=['actual', 'predected'])
                                            st.markdown('Predicted Value')

                                            st.dataframe(coll_pred_actual)

                                    with confusion_metrix_SVM_tab:
                                            # st.metric(label="Accuracy", value=accuracy_SVM)
                                            st.markdown('Report Summary')

                                            st.dataframe(report_SVM)

                                            st.subheader("ROC Curve")
                                    roc1, pre_reca=st.columns(2)
                                    with roc1:
                                        roc_cu_SVM = ETEC.ETEC_SVM.roc_cur(x_train, y_train, x_test, y_test)

                                        st.write('roc curve')
                                        st.pyplot(roc_cu_SVM,clear_figure=True)
                                    with  pre_reca:
                                        pre_recall_SVM = ETEC.ETEC_SVM.preci_recall(x_train, y_train, x_test, y_test)

                                        st.write('precision_recall')
                                        st.pyplot(pre_recall_SVM, clear_figure=True)
                                    pred_error1,pred_error2=st.columns(2)
                                    with pred_error1:
                                        pred_err_SVM = ETEC.ETEC_SVM.predect_error(x_train, y_train, x_test, y_test)

                                        st.write('precision_recall')
                                        st.pyplot(pred_err_SVM, clear_figure=True)

                            with col_SVM3:
                                        Data_model = pd.concat([df[features], df[target]], axis=1)
                                        expander = st.expander('data')
                                        expander.write(Data_model)

                      with KNN:
                            col_KNN1, col_KNN2, col_KNN3 = st.columns((1, 1, 2))
                            with col_KNN1:
                                Weight_KNN = st.selectbox('Weight function',
                                                          ['uniform', 'distance'], key='KNN_Weight')
                                st.session_state['Weight_KNN'] = Weight_KNN
                                algorithm_KNN = st.selectbox('Algorithm for computing the nearest neighbors', ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                                         key='KNN_algorithm')
                                st.session_state['algorithm_KNN'] = algorithm_KNN
                            with col_KNN2:
                                KNN_Leaf_size = st.slider('Leaf size passed to BallTree or KDTree:', 1,100,value=st.session_state['KNN_Leaf_size'],
                                                                   key='Leaf_size_KNN')
                                st.session_state['KNN_Leaf_size'] = KNN_Leaf_size
                                neighbors_KNN = st.slider('Number of neighbors:', 0, 10, st.session_state['neighbors_KNN'],
                                                          key='KNN_neighbors')
                                st.session_state['neighbors_KNN'] = neighbors_KNN

                                # st.button('Build Model', key='build_KNN')
                            if st.button('Build Model', key='KNN_BUILD'):
                                x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                         Train_Test_Split_dt / 100)
                                KNN_Model = ETEC.ETEC_KNN.Train(x_train, y_train,
                                    ETEC_weights=Weight_KNN,
                                    ETEC_n_neighbors=neighbors_KNN,
                                    ETEC_leaf_size=KNN_Leaf_size,
                                ETEC_algorithm=algorithm_KNN)
                                prediction_KNN = ETEC.ETEC_KNN.Prediction(KNN_Model, x_test)
                                accuracy_KNN = ETEC.ETEC_KNN.Accuracy(y_test, prediction_KNN)
                                Confusion_metrix_KNN = ETEC.ETEC_KNN.Accuracy(y_test, prediction_KNN)
                                report_KNN = ETEC.ETEC_KNN.Report(y_test, prediction_KNN)
                                visualization = ETEC.ETEC_KNN.Visualization(y_test, prediction_KNN)
                                predicted_value_KNN_tab, Accuracy_KNN_tab = st.columns([1, 1])

                                with predicted_value_KNN_tab:
                                        fig = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=accuracy_KNN,
                                            title={'text': "Accuracy"},
                                            domain={'x': [0, .6], 'y': [.1, 0.9]}
                                        ))
                                        st.plotly_chart(fig)

                                with Accuracy_KNN_tab:

                                        st.set_option('deprecation.showPyplotGlobalUse', False)
                                        st.markdown('Confusion Metrix')
                                        st.pyplot(visualization, clear_figure=True)

                                vis_acc, confusion_metrix_KNN_tab = st.columns([0.5, 0.9])
                                with vis_acc:

                                        y = y_test.to_numpy()
                                        coll_pred_actual = np.vstack([y, prediction_KNN])
                                        coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                        columns=['actual', 'predected'])
                                        st.markdown('Predicted Value')

                                        st.dataframe(coll_pred_actual)

                                with confusion_metrix_KNN_tab:
                                        # st.metric(label="Accuracy", value=accuracy_KNN)
                                        st.markdown('Report Summary')

                                        st.dataframe(report_KNN)

                                        st.subheader("ROC Curve")
                                roc1, pre_reca = st.columns(2)
                                with roc1:
                                    roc_cu_KNN = ETEC.ETEC_KNN.roc_cur(x_train, y_train, x_test, y_test)

                                    st.write('roc curve')
                                    st.pyplot(roc_cu_KNN, clear_figure=True)
                                with  pre_reca:
                                    pre_recall_KNN = ETEC.ETEC_KNN.preci_recall(x_train, y_train, x_test,
                                                                                          y_test)

                                    st.write('precision_recall')
                                    st.pyplot(pre_recall_KNN, clear_figure=True)
                                pred_error1, pred_error2 = st.columns(2)
                                with pred_error1:
                                    pred_err_KNN = ETEC.ETEC_KNN.predect_error(x_train, y_train, x_test,
                                                                                         y_test)

                                    st.write('precision_recall')
                                    st.pyplot(pred_err_KNN, clear_figure=True)
                                with pred_error2:
                                    cluster_KNN = ETEC.ETEC_KNN.cluster(x_train)

                                    st.write('precision_recall')
                                    st.pyplot(cluster_KNN, clear_figure=True)
                                    # st.write(KNN_Model)
                                    # fig= plot_roc_curve(KNN_Model, x_test, y_test)

                                    # st.pyplot(fig)

                                    # st.subheader("Precision-Recall Curve")
                                    # plot_roc_curve(y_test,prediction_KNN)
                                    # st.pyplot()

                            with col_KNN3:
                                 Data_model = pd.concat([df[features], df[target]], axis=1)
                                 st.write(Data_model)
                      with GB:
                              col_GB1, col_GB2, col_GB3 = st.columns((1, 1, 2))
                              with col_GB1:
                                  loss_GB = st.selectbox('loss function',
                                                            [ 'deviance','exponential'], key='GB_loss')
                                  st.session_state['loss_GB'] = loss_GB
                                  criterion_GB = st.selectbox('criterion',
                                                               ['friedman_mse', 'squared_error', 'mse'],
                                                               key='GB_criterion')
                                  st.session_state['criterion_GB'] = criterion_GB
                              with col_GB2:
                                  GB_learning_rate = st.slider('Learning rate shrinks the contribution of each tree:', 0.0, 1.0,
                                                            value=st.session_state['GB_learning_rate'],step=0.1,
                                                            key='learning_rate_GB')
                                  st.session_state['GB_learning_rate'] = GB_learning_rate
                                  n_estimators = st.slider('The number of boosting stages to perform:', 0, 10, st.session_state['n_estimators_GB'],
                                                            key='GB_n_estimators')
                                  st.session_state['n_estimators_GB'] = n_estimators

                                  # st.button('Build Model', key='build_GB')
                              if st.button('Build Model', key='GB_BUILD'):
                                  x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                           Train_Test_Split_dt / 100)
                                  GB_Model = ETEC.ETEC_Gradient_Boosting.Train(x_train, y_train,
                                                                  ETEC_loss=loss_GB,
                                                                  ETEC_criterion=criterion_GB,
                                                                  ETEC_learning_rate=GB_learning_rate,
                                                                  ETEC_n_estimators=n_estimators)
                                  prediction_GB = ETEC.ETEC_Gradient_Boosting.Prediction(GB_Model, x_test)
                                  accuracy_GB = ETEC.ETEC_Gradient_Boosting.Accuracy(y_test, prediction_GB)
                                  Confusion_metrix_GB = ETEC.ETEC_Gradient_Boosting.Accuracy(y_test, prediction_GB)
                                  report_GB = ETEC.ETEC_Gradient_Boosting.Report(y_test, prediction_GB)
                                  visualization = ETEC.ETEC_Gradient_Boosting.Visualization(y_test, prediction_GB)
                                  predicted_value_GB_tab, Accuracy_GB_tab = st.columns([1, 1])

                                  with predicted_value_GB_tab:
                                    fig = go.Figure(go.Indicator(
                                      mode="gauge+number",
                                      value=accuracy_GB,
                                      title={'text': "Accuracy"},
                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                  ))
                                    st.plotly_chart(fig)

                                  with Accuracy_GB_tab:
                                      st.set_option('deprecation.showPyplotGlobalUse', False)
                                      st.markdown('Confusion Metrix')
                                      st.pyplot(visualization, clear_figure=True)

                                  vis_acc, confusion_metrix_GB_tab = st.columns([0.5, 0.9])
                                  with vis_acc:
                                      y = y_test.to_numpy()
                                      coll_pred_actual = np.vstack([y, prediction_GB])
                                      coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                      columns=['actual', 'predected'])
                                      st.markdown('Predicted Value')

                                      st.dataframe(coll_pred_actual)

                                  with confusion_metrix_GB_tab:
                                  # st.metric(label="Accuracy", value=accuracy_GB)
                                      st.markdown('Report Summary')

                                      st.dataframe(report_GB)

                                      st.subheader("ROC Curve")
                                  roc1, pre_reca = st.columns(2)
                                  with roc1:
                                      roc_cu_GB = ETEC.ETEC_Gradient_Boosting.roc_cur(x_train, y_train, x_test, y_test)

                                      st.write('roc curve')
                                      st.pyplot(roc_cu_GB, clear_figure=True)
                                  with  pre_reca:
                                      pre_recall_GB = ETEC.ETEC_Gradient_Boosting.preci_recall(x_train, y_train, x_test, y_test)

                                      st.write('precision_recall')
                                      st.pyplot(pre_recall_GB, clear_figure=True)
                                  pred_error1, pred_error2 = st.columns(2)
                                  with pred_error1:
                                      pred_err_GB = ETEC.ETEC_Gradient_Boosting.predect_error(x_train, y_train, x_test, y_test)

                                      st.write('precision_recall')
                                      st.pyplot(pred_err_GB, clear_figure=True)

                                      # st.write(GB_Model)
                                      # fig= plot_roc_curve(GB_Model, x_test, y_test)

                                      # st.pyplot(fig)

                                      # st.subheader("Precision-Recall Curve")
                                      # plot_roc_curve(y_test,prediction_GB)
                                      # st.pyplot()

                              with col_GB3:
                                      Data_model = pd.concat([df[features], df[target]], axis=1)
                                      st.write(Data_model)
                      with AB:
                              col_AB1, col_AB2, col_AB3 = st.columns((1, 1, 2))
                              with col_AB1:
                                  algorithm_AB = st.selectbox('algorithm',
                                                         ['SAMME', 'SAMME.R'], key='AB_algorithm')
                                  st.session_state['algorithm_AB'] = algorithm_AB
                                  AB_learning_rate = st.slider('Learning rate (Weight applied to each classifier at each boosting iteration):', 0.0, 1.0,
                                                               value=st.session_state['AB_learning_rate'],
                                                               key='learning_rate_AB')
                                  st.session_state['AB_learning_rate'] = AB_learning_rate
                                  # criterion_AB = st.selectbox('criterion',
                                  #                             ['friedman_mse', 'squared_error', 'mse'],
                                  #                             key='AB_criterion')
                                  # st.session_state['criterion_AB'] = criterion_AB
                              with col_AB2:

                                  n_estimators_AB = st.slider('The maximum number of estimators at which boosting is terminated:', 0, 100,
                                                           st.session_state['n_estimators_AB'],
                                                           key='AB_n_estimators')
                                  st.session_state['n_estimators_AB'] = n_estimators_AB

                                  # st.button('Build Model', key='build_AB')
                              if st.button('Build Model', key='AB_BUILD'):
                                  x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                           Train_Test_Split_dt / 100)
                                  AB_Model = ETEC.ETEC_AdaBoosting.Train(x_train, y_train,
                                                                               ETEC_Y_algorithm=algorithm_AB,
                                                                               #ETEC_criterion=c,
                                                                               ETEC_Y_learning_rate=AB_learning_rate,
                                                                               ETEC_Y_n_estimators=n_estimators_AB)
                                  prediction_AB = ETEC.ETEC_AdaBoosting.Prediction(AB_Model, x_test)
                                  accuracy_AB = ETEC.ETEC_AdaBoosting.Accuracy(y_test, prediction_AB)
                                  Confusion_metrix_AB = ETEC.ETEC_AdaBoosting.Accuracy(y_test, prediction_AB)
                                  report_AB = ETEC.ETEC_AdaBoosting.Report(y_test, prediction_AB)
                                  visualization = ETEC.ETEC_AdaBoosting.Visualization(y_test, prediction_AB)
                                  predicted_value_AB_tab, Accuracy_AB_tab = st.columns([1, 1])

                                  with predicted_value_AB_tab:
                                    fig = go.Figure(go.Indicator(
                                      mode="gauge+number",
                                      value=accuracy_AB,
                                      title={'text': "Accuracy"},
                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                  ))
                                    st.plotly_chart(fig)

                                  with Accuracy_AB_tab:
                                      st.set_option('deprecation.showPyplotGlobalUse', False)
                                      st.markdown('Confusion Metrix')
                                      st.pyplot(visualization, clear_figure=True)

                                  vis_acc, confusion_metrix_AB_tab = st.columns([0.5, 0.9])
                                  with vis_acc:
                                      y = y_test.to_numpy()
                                      coll_pred_actual = np.vstack([y, prediction_AB])
                                      coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                      columns=['actual', 'predected'])
                                      st.markdown('Predicted Value')

                                      st.dataframe(coll_pred_actual)

                                  with confusion_metrix_AB_tab:
                                  # st.metric(label="Accuracy", value=accuracy_AB)
                                      st.markdown('Report Summary')

                                      st.dataframe(report_AB)

                                      st.subheader("ROC Curve")
                                  roc1, pre_reca = st.columns(2)
                                  with roc1:
                                      roc_cu_AB = ETEC.ETEC_AdaBoosting.roc_cur(x_train, y_train, x_test, y_test)

                                      st.write('roc curve')
                                      st.pyplot(roc_cu_AB, clear_figure=True)
                                  with  pre_reca:
                                      pre_recall_AB = ETEC.ETEC_AdaBoosting.preci_recall(x_train, y_train, x_test,
                                                                                               y_test)

                                      st.write('precision_recall')
                                      st.pyplot(pre_recall_AB, clear_figure=True)
                                  pred_error1, pred_error2 = st.columns(2)
                                  with pred_error1:
                                      pred_err_AB = ETEC.ETEC_AdaBoosting.predect_error(x_train, y_train, x_test,
                                                                                              y_test)

                                      st.write('precision_recall')
                                      st.pyplot(pred_err_AB, clear_figure=True)

                                      # st.write(AB_Model)
                                      # fig= plot_roc_curve(AB_Model, x_test, y_test)

                                      # st.pyplot(fig)

                                      # st.subheader("Precision-Recall Curve")
                                      # plot_roc_curve(y_test,prediction_AB)
                                      # st.pyplot()

                              with col_AB3:
                                      Data_model = pd.concat([df[features], df[target]], axis=1)
                                      st.write(Data_model)
                      with NN:
                              col_NN1, col_NN2, col_NN3 = st.columns((1, 1, 2))
                              with col_NN1:
                                  activation_NN = st.selectbox('activation function',
                                                         ['identity', 'logistic','tanh','relu'], key='NN_activation')
                                  st.session_state['activation_NN'] = activation_NN

                                  solver_NN = st.selectbox('solver',
                                                              ['lbfgs', 'sgd', 'adam'],
                                                              key='NN_solver')
                                  st.session_state['solver_NN'] = solver_NN
                              with col_NN2:
                                  NN_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                                                               value=st.session_state['NN_hidden_layer_sizes'],
                                                               key='hidden_layer_sizes_NN')
                                  st.session_state['NN_hidden_layer_sizes'] = NN_hidden_layer_sizes

                                  max_iter_NN = st.slider('Maximum number of iterations:', 200, 1000,
                                                           st.session_state['max_iter_NN'],
                                                           key='NN_max_iter')
                                  st.session_state['max_iter_NN'] = max_iter_NN



                                  # st.button('Build Model', key='build_NN')
                              if st.button('Build Model', key='NN_BUILD'):
                                  x_train, x_test, y_train, y_test = ETEC.ETEC_DATA.Esplit(Data_model, target,
                                                                                           Train_Test_Split_dt / 100)
                                  NN_Model = ETEC.ETEC_Neural_network.Train(x_train, y_train,
                                                                               ETEC_solver=solver_NN,
                                                                               ETEC_max_iter=max_iter_NN,
                                                                               ETEC_hidden_layer_sizes=NN_hidden_layer_sizes,
                                                                               ETEC_activation=activation_NN)
                                  prediction_NN = ETEC.ETEC_Neural_network.Prediction(NN_Model, x_test)
                                  accuracy_NN = ETEC.ETEC_Neural_network.Accuracy(y_test, prediction_NN)
                                  Confusion_metrix_NN = ETEC.ETEC_Neural_network.Accuracy(y_test, prediction_NN)
                                  report_NN = ETEC.ETEC_Neural_network.Report(y_test, prediction_NN)
                                  visualization = ETEC.ETEC_Neural_network.Visualization(y_test, prediction_NN)
                                  predicted_value_NN_tab, Accuracy_NN_tab = st.columns([1, 1])

                                  with predicted_value_NN_tab:
                                   fig = go.Figure(go.Indicator(
                                      mode="gauge+number",
                                      value=accuracy_NN,
                                      title={'text': "Accuracy"},
                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                  ))
                                   st.plotly_chart(fig)

                                  with Accuracy_NN_tab:
                                      st.set_option('deprecation.showPyplotGlobalUse', False)
                                      st.markdown('Confusion Metrix')
                                      st.pyplot(visualization, clear_figure=True)

                                  vis_acc, confusion_metrix_NN_tab = st.columns([0.5, 0.9])
                                  with vis_acc:
                                      y = y_test.to_numpy()
                                      coll_pred_actual = np.vstack([y, prediction_NN])
                                      coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                      columns=['actual', 'predected'])
                                      st.markdown('Predicted Value')

                                      st.dataframe(coll_pred_actual)

                                  with confusion_metrix_NN_tab:
                                  # st.metric(label="Accuracy", value=accuracy_NN)
                                      st.markdown('Report Summary')

                                      st.dataframe(report_NN)

                                      st.subheader("ROC Curve")

                                  roc1, pre_reca = st.columns(2)
                                  with roc1:
                                      roc_cu_NN = ETEC.ETEC_Neural_network.roc_cur(x_train, y_train, x_test, y_test)

                                      st.write('roc curve')
                                      st.pyplot(roc_cu_NN, clear_figure=True)
                                  with  pre_reca:
                                      pre_recall_NN = ETEC.ETEC_Neural_network.preci_recall(x_train, y_train, x_test,
                                                                                               y_test)

                                      st.write('precision_recall')
                                      st.pyplot(pre_recall_NN, clear_figure=True)
                                  pred_error1, pred_error2 = st.columns(2)
                                  with pred_error1:
                                      pred_err_NN = ETEC.ETEC_Neural_network.predect_error(x_train, y_train, x_test,
                                                                                              y_test)

                                      st.write('precision_recall')
                                      st.pyplot(pred_err_NN, clear_figure=True)

                                      # st.write(NN_Model)
                                      # fig= plot_roc_curve(NN_Model, x_test, y_test)

                                      # st.pyplot(fig)

                                      # st.subheader("Precision-Recall Curve")
                                      # plot_roc_curve(y_test,prediction_NN)
                                      # st.pyplot()

                              with col_NN3:
                                  Data_model = pd.concat([df[features], df[target]], axis=1)
                                  st.write(Data_model)
                      # with Logestic_R:
                      #     col_LOGR1, col_LOGR2, col_LOGR3 = st.columns((1, 1, 2))
                      #     with col_LOGR1:
                      #         penalty_LOGR = st.selectbox('Specify the norm of the penalty.',
                      #                                     ['l1', 'l2', 'elasticnet', 'none'], key='LOGR_penalty')
                      #         st.session_state['penalty_LOGR'] = penalty_LOGR
                      #
                      #         solver_LOGR = st.selectbox('lgorithm to use in the optimization problem.',
                      #                                    ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      #                                    key='LOGR_solver')
                      #         st.session_state['solver_LOGR'] = solver_LOGR
                      #
                      #     with col_LOGR3:
                      #         Data_model = pd.concat([df[features], df[target]], axis=1)
                      #         expander_LOGR = st.expander('data')
                      #         expander_LOGR.write(Data_model)
                      #         # st.write(Data_model)
                      #     with col_LOGR2:
                      #         multi_class_LOGR = st.selectbox('multi_class.',
                      #                                         ['auto', 'multinomial', 'ovr'],
                      #                                         key='LOGR_multi_class')
                      #         st.session_state['multi_class_LOGR'] = multi_class_LOGR
                      #
                      #         max_iter_LOGR = st.slider('Maximum number of iterations:', 200, 1000,
                      #                                   st.session_state['max_iter_LOGR'],
                      #                                   key='LOGR_max_iter')
                      #         st.session_state['max_iter_LOGR'] = max_iter_LOGR
                      #
                      #     # st.button('Build Model', key='build_LOGR')
                      #     if st.button('Build Model', key='build_LOGR'):
                      #         LOGR_Model = ETEC.ETEC_LogisticRegression.Train(x_train, y_train,
                      #                                                         ETEC_max_iter=max_iter_LOGR,
                      #                                                         ETEC_solver=solver_LOGR,
                      #                                                         ETEC_penalty=penalty_LOGR,
                      #                                                         ETEC_multi_class=multi_class_LOGR
                      #                                                         )
                      #         prediction_LOGR = ETEC.ETEC_LogisticRegression.Prediction(LOGR_Model, x_test)
                      #         accuracy_LOGR = ETEC.ETEC_LogisticRegression.Accuracy(y_test, prediction_LOGR)
                      #         Confusion_metrix_LOGR = ETEC.ETEC_LogisticRegression.coef(LOGR_Model)
                      #         report_LOGR = ETEC.ETEC_LogisticRegression.Mean_square_error(y_test, prediction_LOGR)
                      #         visualization_LOGR = ETEC.ETEC_LogisticRegression.visualization(y_test, x_test,
                      #                                                                         prediction_LOGR)
                      #         regression_result_LOGR = ETEC.ETEC_LogisticRegression.regression_results(x_train, y_train)
                      #         predicted_value_LOGR_tab, Accuracy_LOGR_tab = st.columns([1, 1])
                      #
                      #         with predicted_value_LOGR_tab:
                      #             fig2 = go.Figure(go.Indicator(
                      #                 mode="gauge+number",
                      #
                      #                 value=math.ceil(accuracy_LOGR),
                      #                 title={'text': "Accuracy"},
                      #                 domain={'x': [0, .6], 'y': [.1, 0.9]}
                      #             ))
                      #             st.plotly_chart(fig2)
                      #
                      #         with Accuracy_LOGR_tab:
                      #             # st.set_option('deprecation.showPyplotGlobalUse', False)
                      #             st.write('Confusion Metrix')
                      #             st.pyplot(visualization_LOGR)
                      #
                      #         vis_acc, confusion_metrix_LOGR_tab = st.columns([0.5, 0.9])
                      #         with vis_acc:
                      #             y = y_test_R.to_numpy()
                      #             coll_pred_actual = np.vstack([y, prediction_LOGR])
                      #             coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                      #                                             columns=['actual', 'predected'])
                      #             st.markdown('Predicted Value')
                      #
                      #             st.dataframe(coll_pred_actual)
                      #
                      #         with confusion_metrix_LOGR_tab:
                      #             # st.metric(label="Accuracy", value=accuracy_LOGR)
                      #             st.markdown('Report Summary')
                      #
                      #             st.write(regression_result_LR)
                      #
                      #             st.subheader("ROC Curve")

          with regression_tab:
              if file is not None:
                  # st.sidebar.write('hello')
                  features_LR = st.multiselect(
                      'features', df.columns, key='LR_Feature'
                  )
                  coltarget_LR, coltrain_test_split_LR = st.columns(2)
                  with coltarget_LR:
                      target_LR = st.selectbox(
                          'Target Value',
                          df.columns, key='LR_Target')

                  with coltrain_test_split_LR:
                      Train_Test_Split_LR = st.slider('Train Test Split', 0, 100,
                                                      st.session_state['Train_Test_Split_LR'],
                                                      key='LR_Train_Test_Split')
                      st.session_state['Train_Test_Split_LR'] = Train_Test_Split_LR
                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)

                  if len(features_LR) > 0:
                      Linear_R,Random_Forest_R,Neural_R,SVR_R,KNN_R,Ridge_R, poison_R, lasso_R,ElasticNetCV_R,BayesianRidge_R,GaussianProcess_R= st.tabs(
                          ["LinearRegression",'Random_Forest_Regression','Neural Network','SVR','KNeighborsRegressor','Ridge', "PoissonRegressor", 'LassoRegressor','ElasticNetCV','BayesianRidge','GaussianProcess'])

                      with Linear_R:
                              col_LR1, col_LR3 = st.columns((1,  1))
                              with col_LR1:
                                  fit_intercept_LR = st.selectbox('calculate the intercept for this model.',
                                                               ['True', 'False'], key='LR_fit_intercept')
                                  st.session_state['fit_intercept_LR'] = fit_intercept_LR

                                 # normalize_LR = st.selectbox('normalize',
                                                            #  ['True', 'False'],
                                                         #  key='LR_normalize')
                                 # st.session_state['normalize_LR'] = normalize_LR
                              with col_LR3:
                                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                                      expander_LR = st.expander('data')
                                      expander_LR.write(Data_model)

                                      # st.write(Data_model)
                              # with col_LR2:
                              #     LR_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                              #                                       value=st.session_state['LR_hidden_layer_sizes'],
                              #                                       key='hidden_layer_sizes_LR')
                              #     st.session_state['LR_hidden_layer_sizes'] = LR_hidden_layer_sizes
                              #
                              #     max_iter_LR = st.slider('Maximum number of iterations:', 200, 1000,
                              #                             st.session_state['max_iter_LR'],
                              #                             key='LR_max_iter')
                              #     st.session_state['max_iter_LR'] = max_iter_LR

                                  # st.button('Build Model', key='build_LR')
                                      x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model,
                                                                                                       target_LR,
                                                                                                       Train_Test_Split_LR / 100)
                                      LR_Model = ETEC.ETEC_LinearRegression.Train(x_train_R, y_train_R,
                                                                                ETEC_fit_intercept=fit_intercept_LR,
                                                                              #  ETEC_normalize=normalize_LR,
                                                                                )
                                      prediction_LR = ETEC.ETEC_LinearRegression.Prediction(LR_Model, x_test_R)
                                      accuracy_LR = ETEC.ETEC_LinearRegression.Accuracy(y_test_R, prediction_LR)
                                      # Confusion_metrix_LR = ETEC.ETEC_LinearRegression.coef(LR_Model)
                                      report_LR = ETEC.ETEC_LinearRegression.Mean_square_error(y_test_R, prediction_LR)
                                      regression_result_LR=ETEC.ETEC_LinearRegression.regression_results2(x_train_R,y_train_R)
                              if st.button('Build Model', key='LR_BUILD'):

                                  # datavis = sns.pairplot(df)
                                  # st.pyplot(datavis)
                                  predicted_value_LR_tab, Accuracy_LR_tab = st.columns([1, 1])
                                  #
                                  # num = math.ceil(accuracy_LR)
                                  # st.write(num)
                                  with predicted_value_LR_tab:
                                      if accuracy_LR > 80:

                                          fig2 = go.Figure(go.Indicator(
                                              mode="gauge+number",
                                              number={'font': {'color': 'green'}},
                                              gauge={
                                                  'axis': {
                                                      'range': [0, 100]
                                                  },
                                                  'bar': {
                                                      'color': "green"}
                                              },
                                              value=math.ceil(accuracy_LR),
                                              title={'text': "Accuracy"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig2)
                                      else:

                                          fig2 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              gauge={
                                                  'axis': {
                                                      'range': [0, 100],

                                                  },
                                                  'bar': {
                                                      'color': "red"}
                                              },

                                              number={'font': {'color': 'red'}},
                                              # gauge={ 'color': 'red'},
                                              value=math.ceil(accuracy_LR),
                                                              title={'text': "Accuracy"},
                                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                              ))
                                          st.plotly_chart(fig2)

                                  with Accuracy_LR_tab:

                                      # st.set_option('deprecation.showPyplotGlobalUse', False)
                                      visualization_LR = ETEC.ETEC_LinearRegression.visualization(x_train_R, y_train_R,
                                                                                                  x_test_R, y_test_R)

                                      st.write('Risidual vs QQ')
                                      st.pyplot(visualization_LR)

                                      # fig = plt.figure(Residual_LR)
                                      # fig_html=mpld3.fig_to_html(fig)
                                      # components.html(fig_html, height=600)
                                  # st.write('component-component plus residual')
                                  # st.write(Residual_LR3)
                                  vis_acc, confusion_metrix_LR_tab = st.columns([1,1])
                                  with vis_acc:
                                      y = y_test_R.to_numpy()
                                      coll_pred_actual = np.vstack([y, prediction_LR])
                                      coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                      columns=['actual', 'predected'])
                                      st.markdown('Predicted Value')

                                      st.dataframe(coll_pred_actual)
                                      fig4 = go.Figure(go.Indicator(
                                          mode="gauge+number",

                                          value=report_LR,
                                          title={'text': "Mean Square Error"},
                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                      ))
                                      st.plotly_chart(fig4)
                                  with confusion_metrix_LR_tab:
                                      # st.metric(label="Accuracy", value=accuracy_LR)
                                      st.markdown('Report Summary')

                                      st.write(regression_result_LR.summary())
                                      st.write(type(regression_result_LR.summary()))
                                  Residual_LR1=ETEC.ETEC_LinearRegression.regplot1(regression_result_LR)

                                  st.write('partial regression plot')
                                  st.plotly_chart(Residual_LR1)

                                  Residual_LR2=ETEC.ETEC_LinearRegression.regplot2(regression_result_LR)

                                  st.write('component-component plus residual')
                                  st.plotly_chart(Residual_LR2)

                                  # st.write(regression_result_LR.cov_params())

                                  # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                                  # # st.write('xoreg')
                                  # st.plotly_chart(Residual_LR3)
                                  err1,err2=st.columns(2)
                                  with err1:
                                      pred_error=ETEC.ETEC_LinearRegression.prediction_error(x_train_R, y_train_R,x_test_R, y_test_R)
                                      st.pyplot(pred_error)


                      with lasso_R:
                          col_LASSOR1, col_LASSOR2,col_LASSOR3 = st.columns((1,1, 2))
                          with col_LASSOR1:
                              fit_intercept_LASSOR = st.selectbox('calculate the intercept for this model.',
                                                              ['True', 'False'], key='LASSOR_fit_intercept')
                              st.session_state['fit_intercept_LASSOR'] = fit_intercept_LASSOR
                              max_iter_LASSOR = st.slider('Maximum number of iterations:', 1000, 5000,
                                                          st.session_state['max_iter_LASSOR'],
                                                          key='LASSOR_max_iter')
                              st.session_state['max_iter_LASSOR'] = max_iter_LASSOR


                          with col_LASSOR3:
                              Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                              expander_LASSOR = st.expander('data')
                              expander_LASSOR.write(Data_model)
                              # st.write(Data_model)
                          with col_LASSOR2:
                              selection_LASSOR = st.selectbox('selection',
                               ['cyclic', 'random'],
                               key='LASSOR_selection')
                              st.session_state['selection_LASSOR'] = selection_LASSOR


                              # precompute_LASSOR = st.selectbox('Whether to use a precomputed Gram matrix to speed up calculations',
                              #  ['True', 'False'],
                              #  key='LASSOR_precompute')
                              # st.session_state['precompute_LASSOR'] = precompute_LASSOR

                              # st.button('Build Model', key='build_LASSOR')
                          if st.button('Build Model', key='LASSOR_BUILD'):

                                  x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model, target_LR,
                                                                                                   Train_Test_Split_LR / 100)
                                  LASSOR_Model = ETEC.ETEC_LassoRegressor.Train(x_train_R, y_train_R,
                                                                            #  ETEC_precompute=precompute_LASSOR,
                                                                                ETEC_selection=selection_LASSOR,
                                                                                ETEC_fit_intercept=fit_intercept_LASSOR,
                                                                                ETEC_max_iter=max_iter_LASSOR
                                                                              #  ETEC_normalize=normalize_LASSOR,
                                                                              )
                                  prediction_LASSOR = ETEC.ETEC_LassoRegressor.Prediction(LASSOR_Model, x_test_R)
                                  accuracy_LASSOR = ETEC.ETEC_LassoRegressor.Accuracy(y_test_R, prediction_LASSOR)
                                  # Confusion_metrix_LASSOR = ETEC.ETEC_LassoRegressor.coef(LASSOR_Model)
                                  report_LASSOR = ETEC.ETEC_LassoRegressor.Mean_square_error(y_test_R, prediction_LASSOR)
                                  visualization_LASSOR = ETEC.ETEC_LassoRegressor.visualization(x_train_R, y_train_R,x_test_R,y_test_R, )
                                  regression_result_LASSOR = ETEC.ETEC_LassoRegressor.regression_results(x_train_R, y_train_R)
                                  predicted_value_LASSOR_tab, Accuracy_LASSOR_tab = st.columns([1, 1])
                                  #
                                  # num = math.ceil(accuracy_LASSOR)
                                  # st.write(num)
                                  with predicted_value_LASSOR_tab:
                                      if accuracy_LASSOR> 80:

                                          fig2 = go.Figure(go.Indicator(
                                              mode="gauge+number",
                                              number={'font': {'color': 'green'}},
                                              gauge={
                                                  'axis': {
                                                      'range': [0, 100]
                                                  },
                                                  'bar': {
                                                      'color': "green"}
                                              },
                                              value=math.ceil(accuracy_LASSOR),
                                              title={'text': "Accuracy"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig2)
                                      else:

                                          fig2 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              gauge={
                                                  'axis': {
                                                      'range': [0, 100],

                                                  },
                                                  'bar': {
                                                      'color': "red"}
                                              },

                                              number={'font': {'color': 'red'}},
                                              # gauge={ 'color': 'red'},
                                              value=math.ceil(accuracy_LASSOR),
                                                              title={'text': "Accuracy"},
                                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                              ))
                                          st.plotly_chart(fig2)

                                  with Accuracy_LASSOR_tab:
                                      st.set_option('deprecation.showPyplotGlobalUse', False)
                                      st.write('Confusion Metrix')
                                      st.pyplot(visualization_LASSOR)

                                  vis_acc, confusion_metrix_LASSOR_tab = st.columns([1,1])
                                  with vis_acc:
                                      y = y_test_R.to_numpy()
                                      coll_pred_actual = np.vstack([y, prediction_LASSOR])
                                      coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                      columns=['actual', 'predected'])
                                      st.markdown('Predicted Value')

                                      st.dataframe(coll_pred_actual)
                                      fig4 = go.Figure(go.Indicator(
                                          mode="gauge+number",

                                          value=report_LASSOR,
                                          title={'text': "Mean Square Error"},
                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                      ))
                                      st.plotly_chart(fig4)

                                  with confusion_metrix_LASSOR_tab:
                                      # st.metric(label="Accuracy", value=accuracy_LR)
                                      st.markdown('Report Summary')
                                      results_summary=regression_result_LASSOR.summary()
                                      # results_as_html = results_summary

                                      # summ_to_df=pd.read_html(results_summary.tables[1].as_html(), header=0, index_col=0)[0]
                                      # summ_to_df2=pd.read_html(results_summary.tables[2].as_html(), header=0, index_col=0)[0]
                                      # summ_to_df3=pd.read_html(results_summary.tables[0].as_html(), header=0, index_col=0)[0]
                                      # st.dataframe(summ_to_df)
                                      # st.dataframe(summ_to_df2)
                                      # st.dataframe(summ_to_df3)

                                      st.write(regression_result_LASSOR.summary())
                                  Residual_LASSOR1 = ETEC.ETEC_PoissonRegressor.regplot1(regression_result_LASSOR)

                                  st.write('partial regression plot')
                                  st.plotly_chart(Residual_LASSOR1)

                                  Residual_LASSOR2 = ETEC.ETEC_PoissonRegressor.regplot2(regression_result_LASSOR)

                                  st.write('component-component plus residual')
                                  st.plotly_chart(Residual_LASSOR2)

                                  # st.write(regression_result_LR.cov_params())

                                  # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                                  # # st.write('xoreg')
                                  # st.plotly_chart(Residual_LR3)
                                  err1, err2 = st.columns(2)
                                  with err1:
                                    pred_error_LASSOR = ETEC.ETEC_PoissonRegressor.prediction_error(x_train_R, y_train_R,
                                                                                                   x_test_R, y_test_R)
                                    st.pyplot(pred_error_LASSOR)

                      with poison_R:
                          col_POISONR1, col_POISONR2, col_POISONR3 = st.columns((1, 1, 2))
                          # with col_POISONR1:
                          #     fit_intercept_POISONR = st.selectbox('calculate the intercept for this model.',
                          #                                         ['True', 'False'], key='POISONR_fit_intercept')
                          #     st.session_state['fit_intercept_POISONR'] = fit_intercept_POISONR

                          with col_POISONR3:
                              Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                              expander_POISONR = st.expander('data')
                              expander_POISONR.write(Data_model)
                              # st.write(Data_model)
                          with col_POISONR1:
                              max_iter_POISONR = st.slider('Maximum number of iterations:', 100, 1000,
                                                           st.session_state['max_iter_POISONR'],
                                                           key='POISONR_max_iter')
                              st.session_state['max_iter_POISONR'] = max_iter_POISONR
                              # selection_POISONR = st.selectbox('selection',
                              #                                 ['cyclic', 'random'],
                              #                                 key='POISONR_selection')
                              # st.session_state['selection_POISONR'] = selection_POISONR

                              # precompute_POISONR = st.selectbox('Whether to use a precomputed Gram matrix to speed up calculations',
                              #  ['True', 'False'],
                              #  key='POISONR_precompute')
                              # st.session_state['precompute_POISONR'] = precompute_POISONR

                              # st.button('Build Model', key='build_POISONR')
                              if st.button('Build Model', key='PoissonR_BUILD'):

                                  x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model, target_LR,
                                                                                                   Train_Test_Split_LR / 100)
                                  POISONR_Model = ETEC.ETEC_PoissonRegressor.Train(x_train_R, y_train_R,
                                                                                #  ETEC_precompute=precompute_POISONR,
                                                                              #  ETEC_selection=selection_POISONR,
                                                                          #      ETEC_fit_intercept=fit_intercept_POISONR,
                                                                                ETEC_max_iter=max_iter_POISONR
                                                                                #  ETEC_normalize=normalize_POISONR,
                                                                                )
                                  prediction_POISONR = ETEC.ETEC_PoissonRegressor.Prediction(POISONR_Model, x_test_R)
                                  accuracy_POISONR = ETEC.ETEC_PoissonRegressor.Accuracy(y_test_R, prediction_POISONR)
                                  # Confusion_metrix_POISONR = ETEC.ETEC_PoissonRegressor.coef(POISONR_Model)
                                  report_POISONR = ETEC.ETEC_PoissonRegressor.Mean_square_error(y_test_R, prediction_POISONR)
                                  visualization_POISONR = ETEC.ETEC_PoissonRegressor.visualization(x_train_R,y_train_R,x_test_R,y_test_R
                                                                                                )
                                  regression_result_POISONR = ETEC.ETEC_PoissonRegressor.regression_results(x_train_R,
                                                                                                         y_train_R)
                                  predicted_value_POISONR_tab, Accuracy_POISONR_tab = st.columns([1, 1])

                                  with predicted_value_POISONR_tab:
                                      if accuracy_POISONR > 80:

                                          fig2 = go.Figure(go.Indicator(
                                              mode="gauge+number",
                                              number={'font': {'color': 'green'}},
                                              gauge={
                                                  'axis': {
                                                      'range': [0, 100]
                                                  },
                                                  'bar': {
                                                      'color': "green"}
                                              },
                                              value=math.ceil(accuracy_POISONR),
                                              title={'text': "Accuracy"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig2)
                                      else:

                                          fig2 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              gauge={
                                                  'axis': {
                                                      'range': [0, 100],

                                                  },
                                                  'bar': {
                                                      'color': "red"}
                                              },

                                              number={'font': {'color': 'red'}},
                                              # gauge={ 'color': 'red'},
                                              value=math.ceil(accuracy_POISONR),
                                                              title={'text': "Accuracy"},
                                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                              ))
                                          st.plotly_chart(fig2)

                                  with Accuracy_POISONR_tab:
                                      st.set_option('deprecation.showPyplotGlobalUse', False)
                                      st.write('Confusion Metrix')
                                      st.pyplot(visualization_POISONR)

                                  vis_acc, confusion_metrix_POISONR_tab = st.columns([1,1])
                                  with vis_acc:
                                      y = y_test_R.to_numpy()
                                      coll_pred_actual = np.vstack([y, prediction_POISONR])
                                      coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                      columns=['actual', 'predected'])
                                      st.markdown('Predicted Value')

                                      st.dataframe(coll_pred_actual)
                                      fig4 = go.Figure(go.Indicator(
                                          mode="gauge+number",

                                          value=report_POISONR,
                                          title={'text': "Mean Square Error"},
                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                      ))
                                      st.plotly_chart(fig4)

                                  with confusion_metrix_POISONR_tab:
                                          # st.metric(label="Accuracy", value=accuracy_LR)
                                          st.markdown('Report Summary')

                                          st.write(regression_result_POISONR.summary())
                                  Residual_POISONR1=ETEC.ETEC_PoissonRegressor.regplot1(regression_result_POISONR)

                                  st.write('partial regression plot')
                                  st.plotly_chart(Residual_POISONR1)

                                  Residual_POISONR2=ETEC.ETEC_PoissonRegressor.regplot2(regression_result_POISONR)

                                  st.write('component-component plus residual')
                                  st.plotly_chart(Residual_POISONR2)

                                      # st.write(regression_result_LR.cov_params())

                                      # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                                      # # st.write('xoreg')
                                      # st.plotly_chart(Residual_LR3)
                                  err1,err2=st.columns(2)
                                  with err1:
                                          pred_error_POISONR=ETEC.ETEC_PoissonRegressor.prediction_error(x_train_R, y_train_R,x_test_R, y_test_R)
                                          st.pyplot(pred_error_POISONR)

                      with ElasticNetCV_R:
                          col_ElasticNetCV1, col_ElasticNetCV2, col_ElasticNetCV3 = st.columns((1, 1, 2))
                          with col_ElasticNetCV1:
                              fit_intercept_ElasticNetCV = st.selectbox('calculate the intercept for this model.',
                                                                  ['True', 'False'], key='ElasticNetCV_fit_intercept')
                              st.session_state['fit_intercept_ElasticNetCV'] = fit_intercept_ElasticNetCV
                              selection_ElasticNetCV = st.selectbox('selection',
                                                                    ['cyclic', 'random'],
                                                                    key='ElasticNetCV_selection')
                              st.session_state['selection_ElasticNetCV'] = selection_ElasticNetCV
                              # precompute_ElasticNetCV = st.selectbox(
                              # 'Whether to use a precomputed Gram matrix to speed up calculations',
                              # ['True', 'False'],
                              # key='ElasticNetCV_precompute')
                              # st.session_state['precompute_ElasticNetCV'] = precompute_ElasticNetCV
                          with col_ElasticNetCV3:
                              Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                              expander_ElasticNetCV = st.expander('data')
                              expander_ElasticNetCV.write(Data_model)
                              # st.write(Data_model)
                          with col_ElasticNetCV2:
                              max_iter_ElasticNetCV = st.slider('Maximum number of iterations:', 1000, 5000,
                                                           st.session_state['max_iter_ElasticNetCV'],
                                                           key='ElasticNetCV_max_iter')
                              st.session_state['max_iter_ElasticNetCV'] = max_iter_ElasticNetCV




                              # st.button('Build Model', key='build_ElasticNetCV')
                          if st.button('Build Model', key='ElasticN_BUILD'):

                              x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model, target_LR,
                                                                                               Train_Test_Split_LR / 100)
                              ElasticNetCV_Model = ETEC.ETEC_ElasticNetCV_Regressor.Train(x_train_R, y_train_R,
                                                                          #  ETEC_precompute=precompute_ElasticNetCV,
                                                                          ETEC_selection=selection_ElasticNetCV,
                                                                          ETEC_fit_intercept=fit_intercept_ElasticNetCV,
                                                                           ETEC_max_iter=max_iter_ElasticNetCV,
                                                                         #    ETEC_normalize=normalize_ElasticNetCV,
                                                                            )
                              prediction_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.Prediction(ElasticNetCV_Model, x_test_R)
                              accuracy_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.Accuracy(y_test_R, prediction_ElasticNetCV)
                              # Confusion_metrix_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.coef(ElasticNetCV_Model)
                              report_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.Mean_square_error(y_test_R, prediction_ElasticNetCV)
                              visualization_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.visualization(x_train_R,
                                                                                                     y_train_R,x_test_R,y_test_R,)
                              regression_result_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.regression_results(x_train_R,
                                                                                                     y_train_R)
                              predicted_value_ElasticNetCV_tab, Accuracy_ElasticNetCV_tab = st.columns([1, 1])
                              #
                              # num = math.ceil(accuracy_ElasticNetCV)
                              # st.write(num)
                              with predicted_value_ElasticNetCV_tab:
                                  if accuracy_ElasticNetCV > 80:

                                      fig2 = go.Figure(go.Indicator(
                                          mode="gauge+number",
                                          number={'font': {'color': 'green'}},
                                          gauge={
                                              'axis': {
                                                  'range': [0, 100]
                                              },
                                              'bar': {
                                                  'color': "green"}
                                          },
                                          value=math.ceil(accuracy_ElasticNetCV),
                                          title={'text': "Accuracy"},
                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                      ))
                                      st.plotly_chart(fig2)
                                  else:

                                      fig2 = go.Figure(go.Indicator(
                                          mode="gauge+number",

                                          gauge={
                                              'axis': {
                                                  'range': [0, 100],

                                              },
                                              'bar': {
                                                  'color': "red"}
                                          },

                                          number={'font': {'color': 'red'}},
                                          # gauge={ 'color': 'red'},
                                          value=math.ceil(accuracy_ElasticNetCV),
                                                          title={'text': "Accuracy"},
                                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                          ))
                                      st.plotly_chart(fig2)

                              with Accuracy_ElasticNetCV_tab:
                                  st.set_option('deprecation.showPyplotGlobalUse', False)
                                  st.write('Confusion Metrix')
                                  st.pyplot(visualization_ElasticNetCV)

                              vis_acc, confusion_metrix_ElasticNetCV_tab = st.columns([1,1])
                              with vis_acc:
                                  y = y_test_R.to_numpy()
                                  coll_pred_actual = np.vstack([y, prediction_ElasticNetCV])
                                  coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                  columns=['actual', 'predected'])
                                  st.markdown('Predicted Value')

                                  st.dataframe(coll_pred_actual)
                                  fig4 = go.Figure(go.Indicator(
                                      mode="gauge+number",

                                      value=report_ElasticNetCV,
                                      title={'text': "Mean Square Error"},
                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                  ))
                                  st.plotly_chart(fig4)

                              with confusion_metrix_ElasticNetCV_tab:
                                  st.markdown('Report Summary')

                                  st.write(regression_result_ElasticNetCV.summary())


                              Residual_ElasticNetCV1 = ETEC.ETEC_ElasticNetCV_Regressor.regplot1(regression_result_ElasticNetCV)

                              st.write('partial regression plot')
                              st.plotly_chart(Residual_ElasticNetCV1)

                              Residual_ElasticNetCV2 = ETEC.ETEC_ElasticNetCV_Regressor.regplot2(regression_result_ElasticNetCV)

                              st.write('component-component plus residual')
                              st.plotly_chart(Residual_ElasticNetCV2)

                                  # st.write(regression_result_LR.cov_params())

                                  # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                                  # # st.write('xoreg')
                                  # st.plotly_chart(Residual_LR3)
                              err1, err2 = st.columns(2)
                              with err1:
                                      pred_error_ElasticNetCV = ETEC.ETEC_ElasticNetCV_Regressor.prediction_error(x_train_R, y_train_R,
                                                                                                      x_test_R, y_test_R)
                                      st.pyplot(pred_error_ElasticNetCV)

                      with BayesianRidge_R:
                          col_BayesianRidge1, col_BayesianRidge2, col_BayesianRidge3 = st.columns((1, 1, 2))
                          with col_BayesianRidge1:
                              fit_intercept_BayesianRidge = st.selectbox('calculate the intercept for this model.',
                                                                  ['True', 'False'], key='BayesianRidge_fit_intercept')
                              st.session_state['fit_intercept_BayesianRidge'] = fit_intercept_BayesianRidge

                              # selection_BayesianRidge = st.selectbox('selection',
                              #                                       ['cyclic', 'random'],
                              #                                       key='BayesianRidge_selection')
                              # st.session_state['selection_BayesianRidge'] = selection_BayesianRidge

                              # precompute_BayesianRidge = st.selectbox(
                              # 'Whether to use a precomputed Gram matrix to speed up calculations',
                              # ['True', 'False'],
                              # key='BayesianRidge_precompute')
                              # st.session_state['precompute_BayesianRidge'] = precompute_BayesianRidge
                          with col_BayesianRidge3:
                              Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                              expander_BayesianRidge = st.expander('data')
                              expander_BayesianRidge.write(Data_model)
                              # st.write(Data_model)
                          with col_BayesianRidge2:
                              max_iter_BayesianRidge = st.slider('Maximum number of iterations:', 300, 1000,
                                                           st.session_state['max_iter_BayesianRidge'],
                                                           key='BayesianRidge_max_iter')
                              st.session_state['max_iter_BayesianRidge'] = max_iter_BayesianRidge




                              # st.button('Build Model', key='build_BayesianRidge')
                          if st.button('Build Model', key='BayesianR_BUILD'):

                              x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model, target_LR,
                                                                                               Train_Test_Split_LR / 100)
                              BayesianRidge_Model = ETEC.ETEC_BayesianRidge_Regressor.Train(x_train_R, y_train_R,
                                                                          #  ETEC_precompute=precompute_BayesianRidge,
                                                                          #ETEC_selection=selection_BayesianRidge,
                                                                          ETEC_fit_intercept=fit_intercept_BayesianRidge,
                                                                           ETEC_n_iter=max_iter_BayesianRidge,
                                                                         #    ETEC_normalize=normalize_BayesianRidge,
                                                                            )
                              prediction_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.Prediction(BayesianRidge_Model, x_test_R)
                              accuracy_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.Accuracy(y_test_R, prediction_BayesianRidge)
                              # Confusion_metrix_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.coef(BayesianRidge_Model)
                              report_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.Mean_square_error(y_test_R, prediction_BayesianRidge)
                              visualization_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.visualization(x_train_R,
                                                                                                     y_train_R,x_test_R,y_test_R,
                                                                                            )
                              regression_result_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.regression_results(x_train_R,
                                                                                                     y_train_R)
                              predicted_value_BayesianRidge_tab, Accuracy_BayesianRidge_tab = st.columns([1, 1])
                              #
                              # num = math.ceil(accuracy_BayesianRidge)
                              # st.write(num)
                              with predicted_value_BayesianRidge_tab:
                                  if accuracy_BayesianRidge > 80:

                                      fig2 = go.Figure(go.Indicator(
                                          mode="gauge+number",
                                          number={'font': {'color': 'green'}},
                                          gauge={
                                              'axis': {
                                                  'range': [0, 100]
                                              },
                                              'bar': {
                                                  'color': "green"}
                                          },
                                          value=math.ceil(accuracy_BayesianRidge),
                                          title={'text': "Accuracy"},
                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                      ))
                                      st.plotly_chart(fig2)
                                  else:

                                      fig2 = go.Figure(go.Indicator(
                                          mode="gauge+number",

                                          gauge={
                                              'axis': {
                                                  'range': [0, 100],

                                              },
                                              'bar': {
                                                  'color': "red"}
                                          },

                                          number={'font': {'color': 'red'}},
                                          # gauge={ 'color': 'red'},
                                          value=math.ceil(accuracy_BayesianRidge),
                                                          title={'text': "Accuracy"},
                                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                          ))
                                      st.plotly_chart(fig2)
                              with Accuracy_BayesianRidge_tab:
                                  st.set_option('deprecation.showPyplotGlobalUse', False)
                                  st.write('Confusion Metrix')
                                  st.pyplot(visualization_BayesianRidge)

                              vis_acc, confusion_metrix_BayesianRidge_tab = st.columns([1,1])
                              with vis_acc:
                                  y = y_test_R.to_numpy()
                                  coll_pred_actual = np.vstack([y, prediction_BayesianRidge])
                                  coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                  columns=['actual', 'predected'])
                                  st.markdown('Predicted Value')

                                  st.dataframe(coll_pred_actual)
                                  fig4 = go.Figure(go.Indicator(
                                      mode="gauge+number",

                                      value=report_BayesianRidge,
                                      title={'text': "Mean Square Error"},
                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                  ))
                                  st.plotly_chart(fig4)
                              with confusion_metrix_BayesianRidge_tab:
                                  st.markdown('Report Summary')

                                  st.write(regression_result_BayesianRidge.summary())

                              Residual_BayesianRidge1 = ETEC.ETEC_BayesianRidge_Regressor.regplot1(
                                  regression_result_BayesianRidge)

                              st.write('partial regression plot')
                              st.plotly_chart(Residual_BayesianRidge1)

                              Residual_BayesianRidge2 = ETEC.ETEC_BayesianRidge_Regressor.regplot2(
                                  regression_result_BayesianRidge)

                              st.write('component-component plus residual')
                              st.plotly_chart(Residual_BayesianRidge2)

                              # st.write(regression_result_LR.cov_params())

                              # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                              # # st.write('xoreg')
                              # st.plotly_chart(Residual_LR3)
                              err1, err2 = st.columns(2)
                              with err1:
                                  pred_error_BayesianRidge = ETEC.ETEC_BayesianRidge_Regressor.prediction_error(x_train_R,
                                                                                                              y_train_R,
                                                                                                              x_test_R,
                                                                                                              y_test_R)
                                  st.pyplot(pred_error_BayesianRidge)

                      with GaussianProcess_R:
                          col_GaussianProcess1, col_GaussianProcess2, col_GaussianProcess3 = st.columns((1, 1, 2))
                          with col_GaussianProcess1:
                              # fit_intercept_GaussianProcess = st.selectbox('calculate the intercept for this model.',
                              #                                     ['True', 'False'], key='GaussianProcess_fit_intercept')
                              # st.session_state['fit_intercept_GaussianProcess'] = fit_intercept_GaussianProcess

                              normalize_y_GaussianProcess = st.selectbox('normalize_y',
                                                                    ['cyclic', 'random'],
                                                                    key='GaussianProcess_normalize_y')
                              st.session_state['normalize_y_GaussianProcess'] = normalize_y_GaussianProcess

                              # precompute_GaussianProcess = st.selectbox(
                              # 'Whether to use a precomputed Gram matrix to speed up calculations',
                              # ['True', 'False'],
                              # key='GaussianProcess_precompute')
                              # st.session_state['precompute_GaussianProcess'] = precompute_GaussianProcess
                          with col_GaussianProcess3:
                              Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                              expander_GaussianProcess = st.expander('data')
                              expander_GaussianProcess.write(Data_model)
                              # st.write(Data_model)
                          with col_GaussianProcess2:
                              random_state_GaussianProcess = st.slider('random_state:', 300, 1000,
                                                           st.session_state['random_state_GaussianProcess'],
                                                           key='GaussianProcess_random_state')
                              st.session_state['random_state_GaussianProcess'] = random_state_GaussianProcess



                              # st.button('Build Model', key='build_GaussianProcess')
                          if st.button('Build Model', key='GaussianR_BUILD'):

                              x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model, target_LR,
                                                                                               Train_Test_Split_LR / 100)
                              GaussianProcess_Model = ETEC.ETEC_GaussianProcess_Regressor.Train(x_train_R, y_train_R,
                                                                          #  ETEC_precompute=precompute_GaussianProcess,
                                                                          #ETEC_selection=selection_GaussianProcess,
                                                                          ETEC_random_state=random_state_GaussianProcess,
                                                                           ETEC_normalize_y=normalize_y_GaussianProcess,
                                                                         #    ETEC_normalize=normalize_GaussianProcess,
                                                                            )
                              prediction_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.Prediction(GaussianProcess_Model, x_test_R)
                              accuracy_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.Accuracy(y_test_R, prediction_GaussianProcess)
                              # Confusion_metrix_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.coef(GaussianProcess_Model)
                              report_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.Mean_square_error(y_test_R, prediction_GaussianProcess)
                              visualization_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.visualization(x_train_R,
                                                                                                     y_train_R,x_test_R,y_test_R,
                                                                                            )
                              regression_result_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.regression_results(x_train_R,
                                                                                                     y_train_R)
                              predicted_value_GaussianProcess_tab, Accuracy_GaussianProcess_tab = st.columns([1, 1])
                              #
                              # num = math.ceil(accuracy_GaussianProcess)
                              st.write(accuracy_GaussianProcess)
                              with predicted_value_GaussianProcess_tab:
                                  if accuracy_GaussianProcess > 80:

                                      fig2 = go.Figure(go.Indicator(
                                          mode="gauge+number",
                                          number={'font': {'color': 'green'}},
                                          gauge={
                                              'axis': {
                                                  'range': [0, 100]
                                              },
                                              'bar': {
                                                  'color': "green"}
                                          },
                                          value=math.ceil(accuracy_GaussianProcess),
                                          title={'text': "Accuracy"},
                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                      ))
                                      st.plotly_chart(fig2)
                                  else:

                                      fig2 = go.Figure(go.Indicator(
                                          mode="gauge+number",

                                          gauge={
                                              'axis': {
                                                  'range': [0, 100],

                                              },
                                              'bar': {
                                                  'color': "red"}
                                          },

                                          number={'font': {'color': 'red'}},
                                          # gauge={ 'color': 'red'},
                                          value=math.ceil(accuracy_GaussianProcess),
                                                          title={'text': "Accuracy"},
                                                          domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                          ))
                                      st.plotly_chart(fig2)

                              with Accuracy_GaussianProcess_tab:
                                  st.set_option('deprecation.showPyplotGlobalUse', False)
                                  st.write('Confusion Metrix')
                                  st.pyplot(visualization_GaussianProcess)

                              vis_acc, confusion_metrix_GaussianProcess_tab = st.columns([1,1])
                              with vis_acc:
                                  y = y_test_R.to_numpy()
                                  coll_pred_actual = np.vstack([y, prediction_GaussianProcess])
                                  coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                  columns=['actual', 'predected'])
                                  st.markdown('Predicted Value')

                                  st.dataframe(coll_pred_actual)
                                  fig4 = go.Figure(go.Indicator(
                                      mode="gauge+number",

                                      value=report_GaussianProcess,
                                      title={'text': "Mean Square Error"},
                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                  ))
                                  st.plotly_chart(fig4)
                                  # st.write(report_GaussianProcess)

                              with confusion_metrix_GaussianProcess_tab:
                                  st.markdown('Report Summary')

                                  st.write(regression_result_GaussianProcess.summary())

                              Residual_GaussianProcess1 = ETEC.ETEC_GaussianProcess_Regressor.regplot1(
                                  regression_result_GaussianProcess)

                              st.write('partial regression plot')
                              st.plotly_chart(Residual_GaussianProcess1)

                              Residual_GaussianProcess2 = ETEC.ETEC_GaussianProcess_Regressor.regplot2(
                                  regression_result_GaussianProcess)

                              st.write('component-component plus residual')
                              st.plotly_chart(Residual_GaussianProcess2)

                              # st.write(regression_result_LR.cov_params())

                              # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                              # # st.write('xoreg')
                              # st.plotly_chart(Residual_LR3)
                              err1, err2 = st.columns(2)
                              with err1:
                                  pred_error_GaussianProcess = ETEC.ETEC_GaussianProcess_Regressor.prediction_error(x_train_R,
                                                                                                              y_train_R,
                                                                                                              x_test_R,
                                                                                                              y_test_R)
                                  st.pyplot(pred_error_GaussianProcess)

                      with Random_Forest_R:
                              col_Random_Forest_R1, col_Random_Forest_R2,col_Random_Forest_R3 = st.columns((1, 1,2))
                              with col_Random_Forest_R1:
                                  criterion_Random_Forest_R = st.selectbox('criterion(quality of a split):',
                                                               ['poisson', 'absolute_error','squared_error'], key='Random_Forest_R_criterion')
                                  st.session_state['criterion_Random_Forest_R'] = criterion_Random_Forest_R

                                  max_features_Random_Forest_R = st.selectbox('max_features',
                                                             ['sqrt', 'log2','None','1.0'],
                                                          key='Random_Forest_R_max_features')
                                  st.session_state['max_features_Random_Forest_R'] = max_features_Random_Forest_R
                              with col_Random_Forest_R2:
                                  n_estimators_Random_Forest_R = st.slider('The number of trees in the forest:', 100, 500,
                                                              st.session_state['n_estimators_Random_Forest_R'],
                                                              key='Random_Forest_R_n_estimators')
                                  st.session_state['n_estimators_Random_Forest_R'] = n_estimators_Random_Forest_R

                                  min_samples_split_Random_Forest_R = st.slider('The minimum number of samples required to split an internal node:', 2,
                                                                           15,
                                                                           st.session_state['min_samples_split_Random_Forest_R'],
                                                                           key='Random_Forest_R_min_samples_split')
                                  st.session_state['min_samples_split_Random_Forest_R'] = min_samples_split_Random_Forest_R
                              with col_Random_Forest_R3:
                                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                                      expander_Random_Forest_R = st.expander('data')
                                      expander_Random_Forest_R.write(Data_model)
                                      # st.write(Data_model)
                              # with col_Random_Forest_R2:
                              #     Random_Forest_R_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                              #                                       value=st.session_state['Random_Forest_R_hidden_layer_sizes'],
                              #                                       key='hidden_layer_sizes_Random_Forest_R')
                              #     st.session_state['Random_Forest_R_hidden_layer_sizes'] = Random_Forest_R_hidden_layer_sizes
                              #
                              #     max_iter_Random_Forest_R = st.slider('Maximum number of iterations:', 200, 1000,
                              #                             st.session_state['max_iter_Random_Forest_R'],
                              #                             key='Random_Forest_R_max_iter')
                              #     st.session_state['max_iter_Random_Forest_R'] = max_iter_Random_Forest_R

                                  # st.button('Build Model', key='build_Random_Forest_R')
                              if st.button('Build Model', key='RFR_BUILD'):

                                      x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model,
                                                                                                       target_LR,
                                                                                                       Train_Test_Split_LR / 100)
                                      Random_Forest_R_Model = ETEC.ETEC_Random_forest_reg.Train(x_train_R, y_train_R,
                                                                                ETEC_criterion=criterion_Random_Forest_R,
                                                                                ETEC_min_samples_split=min_samples_split_Random_Forest_R ,
                                                                                ETEC_max_features=max_features_Random_Forest_R,
                                                                               ETEC_n_estimators=n_estimators_Random_Forest_R,
                                                                                )
                                      prediction_Random_Forest_R = ETEC.ETEC_Random_forest_reg.Prediction(Random_Forest_R_Model, x_test_R)
                                      accuracy_Random_Forest_R = ETEC.ETEC_Random_forest_reg.Accuracy(y_test_R, prediction_Random_Forest_R)
                                      # Confusion_metrix_Random_Forest_R = ETEC.ETEC_Random_forest_reg.coef(Random_Forest_R_Model)
                                      report_Random_Forest_R = ETEC.ETEC_Random_forest_reg.Mean_square_error(y_test_R, prediction_Random_Forest_R)
                                      visualization_Random_Forest_R = ETEC.ETEC_Random_forest_reg.visualization(x_train_R,y_train_R,x_test_R,y_test_R,)
                                      regression_result_Random_Forest_R=ETEC.ETEC_Random_forest_reg.regression_results(x_train_R,y_train_R)

                                      predicted_value_Random_Forest_R_tab, Accuracy_Random_Forest_R_tab = st.columns([1, 1])
                                      #
                                      # num = math.ceil(accuracy_Random_Forest_R)
                                      # st.write(num)
                                      with predicted_value_Random_Forest_R_tab:
                                          if accuracy_Random_Forest_R > 80:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",
                                                  number={'font': {'color': 'green'}},
                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100]
                                                      },
                                                      'bar': {
                                                          'color': "green"}
                                                  },
                                                  value=math.ceil(accuracy_Random_Forest_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)
                                          else:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",

                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100],

                                                      },
                                                      'bar': {
                                                          'color': "red"}
                                                  },

                                                  number={'font': {'color': 'red'}},
                                                  # gauge={ 'color': 'red'},
                                                  value=math.ceil(accuracy_Random_Forest_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)

                                      with Accuracy_Random_Forest_R_tab:
                                          # st.set_option('deprecation.showPyplotGlobalUse', False)
                                          st.write('Confusion Metrix')
                                          st.pyplot(visualization_Random_Forest_R)

                                      vis_acc, confusion_metrix_Random_Forest_R_tab = st.columns([1, 1])
                                      with vis_acc:
                                          y = y_test_R.to_numpy()
                                          coll_pred_actual = np.vstack([y, prediction_Random_Forest_R])
                                          coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                          columns=['actual', 'predected'])
                                          st.markdown('Predicted Value')

                                          st.dataframe(coll_pred_actual)
                                          fig3 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              value=report_Random_Forest_R,
                                              title={'text': "Mean Square Error"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig3)
                                          # st.write(report_Random_Forest_R)

                                      with confusion_metrix_Random_Forest_R_tab:
                                          st.markdown('Report Summary')

                                          st.write(regression_result_Random_Forest_R.summary())

                                      Residual_Random_Forest_R1 = ETEC.ETEC_Random_forest_reg.regplot1(
                                          regression_result_Random_Forest_R)

                                      st.write('partial regression plot')
                                      st.plotly_chart(Residual_Random_Forest_R1)

                                      Residual_Random_Forest_R2 = ETEC.ETEC_Random_forest_reg.regplot2(
                                          regression_result_Random_Forest_R)

                                      st.write('component-component plus residual')
                                      st.plotly_chart(Residual_Random_Forest_R2)

                                      # st.write(regression_result_LR.cov_params())

                                      # Residual_LR3=ETEC.ETEC_LinearRegression.regplot3(regression_result_LR)
                                      # # st.write('xoreg')
                                      # st.plotly_chart(Residual_LR3)
                                      err1, err2 = st.columns(2)
                                      with err1:
                                          pred_error_Random_Forest_R = ETEC.ETEC_Random_forest_reg.prediction_error(
                                              x_train_R,
                                              y_train_R,
                                              x_test_R,
                                              y_test_R)
                                          st.pyplot(pred_error_Random_Forest_R)

                      with Ridge_R:
                              col_Ridge_R2,col_Ridge_R3 = st.columns(( 1,1))

                              with col_Ridge_R2:
                                  alpha_Ridge_R = st.slider('alpha:', 0.0, 1.0,
                                                              st.session_state['alpha_Ridge_R'],
                                                              key='Ridge_R_alpha')
                                  st.session_state['alpha_Ridge_R'] = alpha_Ridge_R


                              with col_Ridge_R3:
                                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                                      expander_Ridge_R = st.expander('data')
                                      expander_Ridge_R.write(Data_model)
                                      # st.write(Data_model)
                              # with col_Ridge_R2:
                              #     Ridge_R_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                              #                                       value=st.session_state['Ridge_R_hidden_layer_sizes'],
                              #                                       key='hidden_layer_sizes_Ridge_R')
                              #     st.session_state['Ridge_R_hidden_layer_sizes'] = Ridge_R_hidden_layer_sizes
                              #
                              #     max_iter_Ridge_R = st.slider('Maximum number of iterations:', 200, 1000,
                              #                             st.session_state['max_iter_Ridge_R'],
                              #                             key='Ridge_R_max_iter')
                              #     st.session_state['max_iter_Ridge_R'] = max_iter_Ridge_R

                                  # st.button('Build Model', key='build_Ridge_R')
                              if st.button('Build Model', key='ridgeR_BUILD'):

                                      x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model,
                                                                                                       target_LR,
                                                                                                       Train_Test_Split_LR / 100)
                                      Ridge_R_Model = ETEC.ETEC_ridge_regression.Train(x_train_R, y_train_R,
                                                                                ETEC_alpha=alpha_Ridge_R,
                                                                                )
                                      prediction_Ridge_R = ETEC.ETEC_ridge_regression.Prediction(Ridge_R_Model, x_test_R)
                                      accuracy_Ridge_R = ETEC.ETEC_ridge_regression.Accuracy(y_test_R, prediction_Ridge_R)
                                      # Confusion_metrix_Ridge_R = ETEC.ETEC_ridge_regression.coef(Ridge_R_Model)
                                      report_Ridge_R = ETEC.ETEC_ridge_regression.Mean_square_error(y_test_R, prediction_Ridge_R)
                                      visualization_Ridge_R = ETEC.ETEC_ridge_regression.visualization(x_train_R,y_train_R,x_test_R,y_test_R,)
                                      regression_result_Ridge_R=ETEC.ETEC_ridge_regression.regression_results(x_train_R,y_train_R)
                                      predicted_value_Ridge_R_tab, Accuracy_Ridge_R_tab = st.columns([1, 1])
                                      #
                                      # num = math.ceil(accuracy_Ridge_R)
                                      # st.write(num)
                                      with predicted_value_Ridge_R_tab:
                                          if accuracy_Ridge_R > 80:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",
                                                  number={'font': {'color': 'green'}},
                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100]
                                                      },
                                                      'bar': {
                                                          'color': "green"}
                                                  },
                                                  value=math.ceil(accuracy_Ridge_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)
                                          else:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",

                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100],

                                                      },
                                                      'bar': {
                                                          'color': "red"}
                                                  },

                                                  number={'font': {'color': 'red'}},
                                                  # gauge={ 'color': 'red'},
                                                  value=math.ceil(accuracy_Ridge_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)

                                      with Accuracy_Ridge_R_tab:
                                          # st.set_option('deprecation.showPyplotGlobalUse', False)
                                          st.write('Confusion Metrix')
                                          st.pyplot(visualization_Ridge_R)

                                      vis_acc, confusion_metrix_Ridge_R_tab = st.columns([1, 1])
                                      with vis_acc:
                                          y = y_test_R.to_numpy()
                                          coll_pred_actual = np.vstack([y, prediction_Ridge_R])
                                          coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                          columns=['actual', 'predected'])
                                          st.markdown('Predicted Value')

                                          st.dataframe(coll_pred_actual)
                                          fig3 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              value=report_Ridge_R,
                                              title={'text': "Mean Square Error"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig3)
                                          # st.write(report_Ridge_R)

                                      with confusion_metrix_Ridge_R_tab:
                                          st.markdown('Report Summary')

                                          st.write(regression_result_Ridge_R.summary())

                                      Residual_Ridge_R1 = ETEC.ETEC_ridge_regression.regplot1(
                                          regression_result_Ridge_R)

                                      st.write('partial regression plot')
                                      st.plotly_chart(Residual_Ridge_R1)

                                      Residual_Ridge_R2 = ETEC.ETEC_ridge_regression.regplot2(
                                          regression_result_Ridge_R)

                                      st.write('component-component plus residual')
                                      st.plotly_chart(Residual_Ridge_R2)

                                      err1, err2 = st.columns(2)
                                      with err1:
                                          pred_error_Ridge_R = ETEC.ETEC_ridge_regression.prediction_error(
                                              x_train_R,
                                              y_train_R,
                                              x_test_R,
                                              y_test_R)
                                          st.pyplot(pred_error_Ridge_R)

                      with Neural_R:
                              col_Neural_R1, col_Neural_R2,col_Neural_R3 = st.columns((1, 1,2))
                              with col_Neural_R1:
                                  activation_Neural_R = st.selectbox('activation:',
                                                               ['identity', 'logistic','tanh','relu'], key='Neural_R_activation')
                                  st.session_state['activation_Neural_R'] = activation_Neural_R

                                  solver_Neural_R = st.selectbox('solver',
                                                             ['lbfgs', 'sgd','adam'],
                                                          key='Neural_R_solver')
                                  st.session_state['solver_Neural_R'] = solver_Neural_R

                                  hidden_layer_sizes_Neural_R = st.slider('hidden_layer_sizes:', 100, 1000,
                                                                    st.session_state['hidden_layer_sizes_Neural_R'],
                                                                    key='Neural_R_hidden_layer_sizes')
                                  st.session_state['hidden_layer_sizes_Neural_R'] = hidden_layer_sizes_Neural_R
                              with col_Neural_R2:
                                  # n_estimators_Neural_R = st.slider('The number of trees in the forest:', 100, 500,
                                  #                             st.session_state['n_estimators_Neural_R'],
                                  #                             key='Neural_R_n_estimators')
                                  # st.session_state['n_estimators_Neural_R'] = n_estimators_Neural_R

                                  learning_rate_Neural_R = st.selectbox('learning_rate',
                                                                 ['constant', 'invscaling', 'adaptive'],
                                                                 key='Neural_R_learning_rate')
                                  st.session_state['learning_rate_Neural_R'] = learning_rate_Neural_R

                                  max_iter_Neural_R = st.slider('Maximum number of iterations:', 100,
                                                                           1000,
                                                                           st.session_state['max_iter_Neural_R'],
                                                                           key='Neural_R_max_iter')
                                  st.session_state['max_iter_Neural_R'] = max_iter_Neural_R
                              with col_Neural_R3:
                                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                                      expander_Neural_R = st.expander('data')
                                      expander_Neural_R.write(Data_model)
                                      # st.write(Data_model)
                              # with col_Neural_R2:
                              #     Neural_R_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                              #                                       value=st.session_state['Neural_R_hidden_layer_sizes'],
                              #                                       key='hidden_layer_sizes_Neural_R')
                              #     st.session_state['Neural_R_hidden_layer_sizes'] = Neural_R_hidden_layer_sizes
                              #
                              #     max_iter_Neural_R = st.slider('Maximum number of iterations:', 200, 1000,
                              #                             st.session_state['max_iter_Neural_R'],
                              #                             key='Neural_R_max_iter')
                              #     st.session_state['max_iter_Neural_R'] = max_iter_Neural_R

                                  # st.button('Build Model', key='build_Neural_R')
                              if st.button('Build Model', key='neuralR_BUILD'):

                                      x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model,
                                                                                                       target_LR,
                                                                                                       Train_Test_Split_LR / 100)
                                      # hidden_layer_sizes_Neural_R,solver_Neural_R,activation_Neural_R,learning_rate_Neural_R,max_iter_Neural_R

                                      Neural_R_Model = ETEC.ETEC_neural_regression.Train(x_train_R, y_train_R,
                                                                                ETEC_hidden_layer_sizes=hidden_layer_sizes_Neural_R,
                                                                                ETEC_max_iter=max_iter_Neural_R ,
                                                                                ETEC_solver=solver_Neural_R,
                                                                               ETEC_learning_rate=learning_rate_Neural_R,
                                                                                ETEC_activation=activation_Neural_R
                                                                                )
                                      prediction_Neural_R = ETEC.ETEC_neural_regression.Prediction(Neural_R_Model, x_test_R)
                                      accuracy_Neural_R = ETEC.ETEC_neural_regression.Accuracy(y_test_R, prediction_Neural_R)
                                      # Confusion_metrix_Neural_R = ETEC.ETEC_neural_regression.coef(Neural_R_Model)
                                      # score_Neural_R=ETEC.ETEC_neural_regression.score(Neural_R_Model,x_test_R,y_test_R)
                                      report_Neural_R = ETEC.ETEC_neural_regression.Mean_square_error(y_test_R, prediction_Neural_R)
                                      visualization_Neural_R = ETEC.ETEC_neural_regression.visualization(x_train_R,y_train_R,x_test_R,y_test_R,)
                                      regression_result_Neural_R=ETEC.ETEC_neural_regression.regression_results(x_train_R,y_train_R)
                                      predicted_value_Neural_R_tab, Accuracy_Neural_R_tab = st.columns([1, 1])
                                      #
                                      # num = math.ceil(accuracy_Neural_R)
                                      # st.write(num)
                                      with predicted_value_Neural_R_tab:
                                          if accuracy_Neural_R > 80:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",
                                                  number={'font': {'color': 'green'}},
                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100]
                                                      },
                                                      'bar': {
                                                          'color': "green"}
                                                  },
                                                  value=math.ceil(accuracy_Neural_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)
                                          else:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",

                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100],

                                                      },
                                                      'bar': {
                                                          'color': "red"}
                                                  },

                                                  number={'font': {'color': 'red'}},
                                                  # gauge={ 'color': 'red'},
                                                  value=math.ceil(accuracy_Neural_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)

                                      with Accuracy_Neural_R_tab:
                                          # st.set_option('deprecation.showPyplotGlobalUse', False)
                                          st.write('Confusion Metrix')
                                          st.pyplot(visualization_Neural_R)

                                      vis_acc, confusion_metrix_Neural_R_tab = st.columns([1, 1])
                                      with vis_acc:
                                          y = y_test_R.to_numpy()
                                          coll_pred_actual = np.vstack([y, prediction_Neural_R])
                                          coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                          columns=['actual', 'predected'])
                                          st.markdown('Predicted Value')

                                          st.dataframe(coll_pred_actual)
                                          fig3 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              value=report_Neural_R,
                                              title={'text': "Mean Square Error"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig3)
                                          # st.write(score_Neural_R)

                                      with confusion_metrix_Neural_R_tab:
                                          st.markdown('Report Summary')

                                          st.write(regression_result_Neural_R.summary())

                                      Residual_Neural_R1 = ETEC.ETEC_neural_regression.regplot1(
                                          regression_result_Neural_R)

                                      st.write('partial regression plot')
                                      st.plotly_chart(Residual_Neural_R1)

                                      Residual_Neural_R2 = ETEC.ETEC_neural_regression.regplot2(
                                          regression_result_Neural_R)

                                      st.write('component-component plus residual')
                                      st.plotly_chart(Residual_Neural_R2)

                                      err1, err2 = st.columns(2)
                                      with err1:
                                          pred_error_Neural_R = ETEC.ETEC_neural_regression.prediction_error(
                                              x_train_R,
                                              y_train_R,
                                              x_test_R,
                                              y_test_R)
                                          st.pyplot(pred_error_Neural_R)

                      with SVR_R:
                              col_SVR_R1,col_SVR_R3 = st.columns((1, 1))
                              with col_SVR_R1:
                                  kernel_SVR_R = st.selectbox('kernel:',
                                                               ['linear', 'poly','rbf','sigmoid','precomputed'], key='SVR_R_kernel')
                                  st.session_state['kernel_SVR_R'] = kernel_SVR_R

                                  gamma_SVR_R = st.selectbox('gamma',
                                                             ['scale', 'auto'],
                                                          key='SVR_R_gamma')
                                  st.session_state['gamma_SVR_R'] = gamma_SVR_R

                                  degree_SVR_R = st.slider('Degree of the polynomial kernel function:', 2, 10,
                                                                    st.session_state['degree_SVR_R'],
                                                                    key='SVR_R_degree')
                                  st.session_state['degree_SVR_R'] = degree_SVR_R

                              # with col_SVR_R2:
                              #     # n_estimators_SVR_R = st.slider('The number of trees in the forest:', 100, 500,
                              #     #                             st.session_state['n_estimators_SVR_R'],
                              #     #                             key='SVR_R_n_estimators')
                              #     # st.session_state['n_estimators_SVR_R'] = n_estimators_SVR_R
                              #
                              #     learning_rate_SVR_R = st.selectbox('learning_rate',
                              #                                    ['constant', 'invscaling', 'adaptive'],
                              #                                    key='SVR_R_learning_rate')
                              #     st.session_state['learning_rate_SVR_R'] = learning_rate_SVR_R
                              #
                              #     max_iter_SVR_R = st.slider('Maximum number of iterations:', 100,
                              #                                              1000,
                              #                                              st.session_state['max_iter_SVR_R'],
                              #                                              key='SVR_R_max_iter')
                              #     st.session_state['max_iter_SVR_R'] = max_iter_SVR_R
                              with col_SVR_R3:
                                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                                      expander_SVR_R = st.expander('data')
                                      expander_SVR_R.write(Data_model)
                                      # st.write(Data_model)
                              # with col_SVR_R2:
                              #     SVR_R_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                              #                                       value=st.session_state['SVR_R_hidden_layer_sizes'],
                              #                                       key='hidden_layer_sizes_SVR_R')
                              #     st.session_state['SVR_R_hidden_layer_sizes'] = SVR_R_hidden_layer_sizes
                              #
                              #     max_iter_SVR_R = st.slider('Maximum number of iterations:', 200, 1000,
                              #                             st.session_state['max_iter_SVR_R'],
                              #                             key='SVR_R_max_iter')
                              #     st.session_state['max_iter_SVR_R'] = max_iter_SVR_R

                                  # st.button('Build Model', key='build_SVR_R')
                              if st.button('Build Model', key='SVR_BUILD'):

                                      x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model,
                                                                                                       target_LR,
                                                                                                       Train_Test_Split_LR / 100)
                                      # hidden_layer_sizes_SVR_R,solver_SVR_R,activation_SVR_R,learning_rate_SVR_R,max_iter_SVR_R

                                      SVR_R_Model = ETEC.ETEC_SVR_regression.Train(x_train_R, y_train_R,
                                                                                ETEC_kernel=kernel_SVR_R,
                                                                                ETEC_gamma=gamma_SVR_R ,
                                                                                ETEC_degree=degree_SVR_R,
                                                                               # ETEC_learning_rate=learning_rate_SVR_R,
                                                                               #  ETEC_activation=activation_SVR_R
                                                                                )
                                      prediction_SVR_R = ETEC.ETEC_SVR_regression.Prediction(SVR_R_Model, x_test_R)
                                      accuracy_SVR_R = ETEC.ETEC_SVR_regression.Accuracy(y_test_R, prediction_SVR_R)
                                      # Confusion_metrix_SVR_R = ETEC.ETEC_SVR_regression.coef(SVR_R_Model)
                                      # score_SVR_R=ETEC.ETEC_SVR_regression.score(SVR_R_Model,x_test_R,y_test_R)
                                      report_SVR_R = ETEC.ETEC_SVR_regression.Mean_square_error(y_test_R, prediction_SVR_R)
                                      visualization_SVR_R = ETEC.ETEC_SVR_regression.visualization(x_train_R,y_train_R,x_test_R,y_test_R,)
                                      regression_result_SVR_R=ETEC.ETEC_SVR_regression.regression_results(x_train_R,y_train_R)
                                      predicted_value_SVR_R_tab, Accuracy_SVR_R_tab = st.columns([1, 1])
                                      #
                                      # num = math.ceil(accuracy_SVR_R)
                                      # st.write(num)
                                      with predicted_value_SVR_R_tab:
                                          if accuracy_SVR_R>80:

                                                  fig2 = go.Figure(go.Indicator(
                                                      mode="gauge+number",
                                                      number={'font': {'color': 'green'}},
                                                      gauge={
                                                          'axis': {
                                                              'range': [0, 100]
                                                          },
                                                      'bar': {
                                                          'color': "green"}
                                                  },
                                                      value=math.ceil(accuracy_SVR_R),
                                                      title={'text': "Accuracy"},
                                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                  ))
                                                  st.plotly_chart(fig2)
                                          else:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",

                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100],


                                                      },
                                                      'bar': {
                                                          'color': "red"}
                                                  },

                                                  number={'font': {'color': 'red'}},
                                                  # gauge={ 'color': 'red'},
                                                  value=math.ceil(accuracy_SVR_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)
                                      with Accuracy_SVR_R_tab:
                                          # st.set_option('deprecation.showPyplotGlobalUse', False)
                                          st.write('Confusion Metrix')
                                          st.pyplot(visualization_SVR_R)

                                      vis_acc, confusion_metrix_SVR_R_tab = st.columns([1, 1])
                                      with vis_acc:
                                          y = y_test_R.to_numpy()
                                          coll_pred_actual = np.vstack([y, prediction_SVR_R])
                                          coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                          columns=['actual', 'predected'])
                                          st.markdown('Predicted Value')

                                          st.dataframe(coll_pred_actual)
                                          fig3 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              value=report_SVR_R,
                                              title={'text': "Mean Square Error"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig3)
                                          # st.write(score_SVR_R)

                                      with confusion_metrix_SVR_R_tab:
                                          st.markdown('Report Summary')

                                          st.write(regression_result_SVR_R.summary())

                                      Residual_SVR_R1 = ETEC.ETEC_SVR_regression.regplot1(
                                          regression_result_SVR_R)

                                      st.write('partial regression plot')
                                      st.plotly_chart(Residual_SVR_R1)

                                      Residual_SVR_R2 = ETEC.ETEC_SVR_regression.regplot2(
                                          regression_result_SVR_R)

                                      st.write('component-component plus residual')
                                      st.plotly_chart(Residual_SVR_R2)

                                      err1, err2 = st.columns(2)
                                      with err1:
                                          pred_error_SVR_R = ETEC.ETEC_SVR_regression.prediction_error(
                                              x_train_R,
                                              y_train_R,
                                              x_test_R,
                                              y_test_R)
                                          st.pyplot(pred_error_SVR_R)

                      with KNN_R:
                              col_KNN_R1,col_KNN_R3 = st.columns((1, 1))
                              with col_KNN_R1:
                                  algorithm_KNN_R = st.selectbox('algorithm:',
                                                               ['auto', 'ball_tree','kd_tree','brute'], key='KNN_R_algorithm')
                                  st.session_state['algorithm_KNN_R'] = algorithm_KNN_R

                                  weights_KNN_R = st.selectbox('weights',
                                                             ['uniform', 'distance'],
                                                          key='KNN_R_weights')
                                  st.session_state['weights_KNN_R'] = weights_KNN_R

                                  n_neighbors_KNN_R = st.slider('Number of neighbors :', 2, 10,
                                                                    st.session_state['n_neighbors_KNN_R'],
                                                                    key='KNN_R_n_neighbors')
                                  st.session_state['n_neighbors_KNN_R'] = n_neighbors_KNN_R
                              # with col_KNN_R2:
                              #     # n_estimators_KNN_R = st.slider('The number of trees in the forest:', 100, 500,
                              #     #                             st.session_state['n_estimators_KNN_R'],
                              #     #                             key='KNN_R_n_estimators')
                              #     # st.session_state['n_estimators_KNN_R'] = n_estimators_KNN_R
                              #
                              #     learning_rate_KNN_R = st.selectbox('learning_rate',
                              #                                    ['constant', 'invscaling', 'adaptive'],
                              #                                    key='KNN_R_learning_rate')
                              #     st.session_state['learning_rate_KNN_R'] = learning_rate_KNN_R
                              #
                              #     max_iter_KNN_R = st.slider('Maximum number of iterations:', 100,
                              #                                              1000,
                              #                                              st.session_state['max_iter_KNN_R'],
                              #                                              key='KNN_R_max_iter')
                              #     st.session_state['max_iter_KNN_R'] = max_iter_KNN_R
                              with col_KNN_R3:
                                      Data_model = pd.concat([df[features_LR], df[target_LR]], axis=1)
                                      expander_KNN_R = st.expander('data')
                                      expander_KNN_R.write(Data_model)
                                      # st.write(Data_model)
                              # with col_KNN_R2:
                              #     KNN_R_hidden_layer_sizes = st.slider('hidden_layer_sizes:', 50, 300,
                              #                                       value=st.session_state['KNN_R_hidden_layer_sizes'],
                              #                                       key='hidden_layer_sizes_KNN_R')
                              #     st.session_state['KNN_R_hidden_layer_sizes'] = KNN_R_hidden_layer_sizes
                              #
                              #     max_iter_KNN_R = st.slider('Maximum number of iterations:', 200, 1000,
                              #                             st.session_state['max_iter_KNN_R'],
                              #                             key='KNN_R_max_iter')
                              #     st.session_state['max_iter_KNN_R'] = max_iter_KNN_R

                                  # st.button('Build Model', key='build_KNN_R')
                              if st.button('Build Model', key='KNNR_BUILD'):

                                      x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(Data_model,
                                                                                                       target_LR,
                                                                                                       Train_Test_Split_LR / 100)
                                      # hidden_layer_sizes_KNN_R,solver_KNN_R,activation_KNN_R,learning_rate_KNN_R,max_iter_KNN_R

                                      KNN_R_Model = ETEC.ETEC_KNN_regression.Train(x_train_R, y_train_R,
                                                                                ETEC_weights=weights_KNN_R,
                                                                                ETEC_algorithm=algorithm_KNN_R ,
                                                                                ETEC_n_neighbors=n_neighbors_KNN_R,
                                                                               # ETEC_learning_rate=learning_rate_KNN_R,
                                                                               #  ETEC_activation=activation_KNN_R
                                                                                )
                                      prediction_KNN_R = ETEC.ETEC_KNN_regression.Prediction(KNN_R_Model, x_test_R)
                                      accuracy_KNN_R = ETEC.ETEC_KNN_regression.Accuracy(y_test_R, prediction_KNN_R)
                                      # Confusion_metrix_KNN_R = ETEC.ETEC_KNN_regression.coef(KNN_R_Model)
                                      # score_KNN_R=ETEC.ETEC_KNN_regression.score(KNN_R_Model,x_test_R,y_test_R)
                                      report_KNN_R = ETEC.ETEC_KNN_regression.Mean_square_error(y_test_R, prediction_KNN_R)
                                      visualization_KNN_R = ETEC.ETEC_KNN_regression.visualization(x_train_R,y_train_R,x_test_R,y_test_R,)
                                      regression_result_KNN_R=ETEC.ETEC_KNN_regression.regression_results(x_train_R,y_train_R)
                                      predicted_value_KNN_R_tab, Accuracy_KNN_R_tab = st.columns([1, 1])
                                      #
                                      # num = math.ceil(accuracy_KNN_R)
                                      # st.write(num)
                                      with predicted_value_KNN_R_tab:
                                          if accuracy_KNN_R>80:

                                                  fig2 = go.Figure(go.Indicator(
                                                      mode="gauge+number",
                                                      number={'font': {'color': 'green'}},
                                                      gauge={
                                                          'axis': {
                                                              'range': [0, 100]
                                                          },
                                                      'bar': {
                                                          'color': "green"}
                                                  },
                                                      value=math.ceil(accuracy_KNN_R),
                                                      title={'text': "Accuracy"},
                                                      domain={'x': [0, .6], 'y': [.1, 0.9]}
                                                  ))
                                                  st.plotly_chart(fig2)
                                          else:

                                              fig2 = go.Figure(go.Indicator(
                                                  mode="gauge+number",

                                                  gauge={
                                                      'axis': {
                                                          'range': [0, 100],


                                                      },
                                                      'bar': {
                                                          'color': "red"}
                                                  },

                                                  number={'font': {'color': 'red'}},
                                                  # gauge={ 'color': 'red'},
                                                  value=math.ceil(accuracy_KNN_R),
                                                  title={'text': "Accuracy"},
                                                  domain={'x': [0, .6], 'y': [.1, 0.9]}
                                              ))
                                              st.plotly_chart(fig2)
                                      with Accuracy_KNN_R_tab:
                                          # st.set_option('deprecation.showPyplotGlobalUse', False)
                                          st.write('Confusion Metrix')
                                          st.pyplot(visualization_KNN_R)

                                      vis_acc, confusion_metrix_KNN_R_tab = st.columns([1, 1])
                                      with vis_acc:
                                          y = y_test_R.to_numpy()
                                          coll_pred_actual = np.vstack([y, prediction_KNN_R])
                                          coll_pred_actual = pd.DataFrame(data=np.rot90(coll_pred_actual),
                                                                          columns=['actual', 'predected'])
                                          st.markdown('Predicted Value')

                                          st.dataframe(coll_pred_actual)
                                          fig3 = go.Figure(go.Indicator(
                                              mode="gauge+number",

                                              value=report_KNN_R,
                                              title={'text': "Mean Square Error"},
                                              domain={'x': [0, .6], 'y': [.1, 0.9]}
                                          ))
                                          st.plotly_chart(fig3)
                                          # st.write(score_KNN_R)

                                      with confusion_metrix_KNN_R_tab:
                                          st.markdown('Report Summary')

                                          st.write(regression_result_KNN_R.summary())

                                      Residual_KNN_R1 = ETEC.ETEC_KNN_regression.regplot1(
                                          regression_result_KNN_R)

                                      st.write('partial regression plot')
                                      st.plotly_chart(Residual_KNN_R1)

                                      Residual_KNN_R2 = ETEC.ETEC_KNN_regression.regplot2(
                                          regression_result_KNN_R)

                                      st.write('component-component plus residual')
                                      st.plotly_chart(Residual_KNN_R2)

                                      err1, err2 = st.columns(2)
                                      with err1:
                                          pred_error_KNN_R = ETEC.ETEC_KNN_regression.prediction_error(
                                              x_train_R,
                                              y_train_R,
                                              x_test_R,
                                              y_test_R)
                                          st.pyplot(pred_error_KNN_R)

          with tab3:
            if file is not None:
                try:
                    coltarget_2, colFeature2 = st.columns(2)
                    with coltarget_2:

                        target_L2 = st.selectbox(
                            'Target Value',
                            df.columns, key='OWL_Target')
                    # with colFeature2:
                    #     features = st.multiselect(
                    #         'features', df.columns, key='DT_Feature'
                    #     )
                    # df_copy=df.drop([target_L2],axis=1)
                    # x_train_R, x_test_R, y_train_R, y_test_R = ETEC.ETEC_DATA.Esplit(df,
                    #                                                                  target_L2,
                    #                                                                  Train_Test_Split_LR / 100)
                    df=ETEC.ETEC_DATA.Eclean(df)
                    df = df.dropna()
                    x = sm.add_constant(df.select_dtypes(include=[np.number]))
                    model = sm.OLS(df[target_L2], x).fit()
                    # regression_result = ETEC.ETEC_LinearRegression.regression_results(df.select_dtypes(include=[np.number]).columns, df[target_L2])
                    # st.write(df_copy.select_dtypes(include=[np.number]))

                    st.write(model.summary())
                except:
                    ''

if __name__ == '__main__':
    main()
