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
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import seaborn as sns
import ETEC_Watson_Analysis5 as ETEC
#######   Decission Tree   #######

if 'Train_Test_Split_dt' not in st.session_state:
    st.session_state['Train_Test_Split_dt'] = 20
if 'min_sample_split_dt' not in st.session_state:
    st.session_state['min_sample_split_dt'] = 2
if 'min_sample_leaf_dt' not in st.session_state:
    st.session_state['min_sample_leaf_dt'] = 2
if 'max_leaf_node_dt' not in st.session_state:
    st.session_state['max_leaf_node_dt'] = 0
if 'random_state_dt' not in st.session_state:
    st.session_state['random_state_dt'] = 0
#######   Random_Forest   #######
if 'Train_Test_Split_RF' not in st.session_state:
    st.session_state['Train_Test_Split_RF'] = 20
if 'min_sample_split_rf' not in st.session_state:
    st.session_state['min_sample_split_rf'] = 2
if 'min_sample_leaf_rf' not in st.session_state:
    st.session_state['min_sample_leaf_rf'] = 2
if 'max_leaf_node_rf' not in st.session_state:
    st.session_state['max_leaf_node_rf'] = 0
if 'random_state_rf' not in st.session_state:
    st.session_state['random_state_rf'] = 0
#######   Support Vector Machine   #######
if 'Train_Test_Split_SVM' not in st.session_state:
    st.session_state['Train_Test_Split_SVM'] = 20
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
if 'Train_Test_Split_KNN' not in st.session_state:
    st.session_state['Train_Test_Split_KNN'] = 20
if 'KNN_Leaf_size' not in st.session_state:
    st.session_state['KNN_Leaf_size'] = 30
if 'algorithm_KNN' not in st.session_state:
    st.session_state['algorithm_KNN'] = 'auto'
if 'Weight_KNN' not in st.session_state:
    st.session_state['Weight_KNN'] = 'uniform'
if 'neighbors_KNN' not in st.session_state:
    st.session_state['neighbors_KNN'] = 5
st.set_page_config(layout="wide")
sns.set()
# px.defaults.width = 800
# px.defaults.height = 500


# Functions Exploratory Analysis
class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=[np.object]).columns

    def box_plot(self, main_var, col_x=None, hue=None):
        return px.box(self.df, x=col_x, y=main_var, color=hue)

    def violin(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.violinplot(x=col_x, y=main_var, hue=hue,
                              data=self.df, palette="husl", split=split)

    def swarmplot(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.swarmplot(x=col_x, y=main_var, hue=hue,
                             data=self.df, palette="husl", dodge=split)

    def histogram_num(self, main_var, hue=None, bins=None, ranger=None):
        return px.histogram(self.df[self.df[main_var].between(left=ranger[0], right=ranger[1])], \
                            x=main_var, nbins=bins, color=hue, marginal='violin')

    def scatter_plot(self, col_x, col_y, hue=None, size=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue, size=size)

    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y, color=hue)

    def line_plot(self, col_y, col_x, hue=None, group=None):
        return px.line(self.df, x=col_x, y=col_y, color=hue, line_group=group)

    def CountPlot(self, main_var, hue=None):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.countplot(x=main_var, data=self.df, hue=hue, palette='pastel')
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def heatmap_vars(self, cols, func=np.mean):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(
            self.df.pivot_table(index=cols[0], columns=cols[1], values=cols[2], aggfunc=func, fill_value=0).dropna(
                axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

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

    def DistPlot(self, main_var):
        sns.set(style="whitegrid")
        return sns.distplot(self.df[main_var], color='c', rug=True)


def get_data(file):
    read_cache_csv = st.cache(pd.read_csv, allow_output_mutation=True)
    df = read_cache_csv(file)
    return df


@st.cache
def get_stats(df):
    stats_num = df.describe()
    if df.select_dtypes(np.object).empty:
        return stats_num.transpose(), None
    if df.select_dtypes(np.number).empty:
        return None, df.describe(include=np.object).transpose()
    else:
        return stats_num.transpose(), df.describe(include=np.object).transpose()


@st.cache
def get_info(df):
    return pd.DataFrame(
        {'types': df.dtypes, 'nan': df.isna().sum(), 'nan%': round((df.isna().sum() / len(df)) * 100, 2),
         'unique': df.nunique()})


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


@st.cache
def get_na_info(df_preproc, df, col):
    raw_info = pd_of_stats(df, col)
    prep_info = pd_of_stats(df_preproc, col)
    return raw_info.join(prep_info, lsuffix='_raw', rsuffix='_prep').T


@st.cache
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


@st.cache
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


@st.cache
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


@st.cache
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


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


def main():
    st.title('Exploratory Data Analysis :mag:')
    st.header('Analyze the descriptive statistics and the distribution of your data. Preview and save your graphics.')

    file = st.file_uploader('Upload your file (.csv)', type='csv')


    if file is not None:
        tab1, tab2, tab3 = st.tabs(["Data Summary", "model", "Owl"])
        with tab1:


            df = get_data(file)

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
          if file is not None:
            DT,RF, SVM,KNN,GB,AB,NN = st.tabs(["machine learning", "Random Forest", "Support Vector Machine",'K-nearest Neighbour','Gradiant Boosting','Adaptive Boosting','Neural Network'])


            with DT:
                features = st.multiselect(
                    'features', df.columns,key='DT_Feature'
                )
                colDT1,colDT2=st.columns(2)
                with colDT2:
                    target = st.selectbox(
                        'Target Value',
                        df.columns,key='DT_Target')
                with colDT1:
                    Train_Test_Split_dt = st.slider('Train Test Split', 0, 100, st.session_state['Train_Test_Split_dt'],key='DT_Train_Test_Split')
                    st.session_state['Train_Test_Split_dt'] = Train_Test_Split_dt
                col_DT1,col_DT2,col_DT3=st.columns((1,1,2))
                with col_DT1:
                    min_sample_split_dt = st.slider('min sample split', 0, 100, st.session_state['min_sample_split_dt'],
                                                    key='DT_min_sample_split')
                    st.session_state['min_sample_split_dt'] = min_sample_split_dt

                    min_sample_leaf_dt = st.slider('min sample leaf', 0, 100, st.session_state['min_sample_leaf_dt'])
                    st.session_state['min_sample_leaf_dt'] = min_sample_leaf_dt
                    max_leaf_node_dt = st.slider('max leaf node', 0, 100, st.session_state['max_leaf_node_dt'])
                    st.session_state['max_leaf_node_dt'] = max_leaf_node_dt
                with col_DT2:
                    random_state = st.number_input('random state', value=st.session_state['random_state_dt'])
                    st.session_state['random_state_dt'] = random_state
                    option = st.selectbox(
                        'criteria',
                        ('gini', 'entropy', 'log_loss'))

                    st.button('Build Model')

                with col_DT3:
                    Data_model = pd.concat([df[features], df[target]], axis=1)
                    st.write(Data_model)
            with RF:
                features = st.multiselect(
                    'features', df.columns,key='RF_Feature'
                )
                colRF1, colRF2 = st.columns(2)
                with colRF2:
                    target = st.selectbox(
                        'Target Value',
                        df.columns,key='RF_Target')
                with colRF1:
                    Train_Test_Split_RF = st.slider('Train Test Split', 0, 100, st.session_state['Train_Test_Split_RF'],
                                                    key='RF_Train_Test_Split')
                    st.session_state['Train_Test_Split_RF'] = Train_Test_Split_RF

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

                    st.button('Build Model',key='build_rf')
                # if st.button('Build Model'):
                #     ETEC.ETEC_Decission_Tree()
            with SVM:
                features = st.multiselect(
                    'features', df.columns, key='SVM_Feature'
                )
                colSVM1, colSVM2 = st.columns(2)
                with colSVM2:
                    target = st.selectbox(
                        'Target Value',
                        df.columns, key='SVM_Target')
                with colSVM1:
                    Train_Test_Split_SVM = st.slider('Train Test Split', 0, 100, st.session_state['Train_Test_Split_SVM'],
                                                    key='SVM_Train_Test_Split')
                    st.session_state['Train_Test_Split_SVM'] = Train_Test_Split_SVM

                col_SVM1, col_SVM2, col_SVM3 = st.columns((1, 1, 2))
                with col_SVM1:
                    kernal_SVM = st.selectbox('Select the Kernel coefficient for rbf, poly and sigmoid',['kernal','scale'] ,key='SVM_kernal')
                    st.session_state['kernal_SVM'] = kernal_SVM
                    gamma_SVM = st.selectbox('gamma', ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
                                              key='SVM_gamma')
                    st.session_state['gamma_SVM'] = gamma_SVM
                with col_SVM2:
                    SVM_random_state = st.number_input('random state', value=st.session_state['SVM_random_state'],key='random_state_SVM')
                    st.session_state['SVM_random_state'] = SVM_random_state
                    Degree_SVM = st.slider('Degree of the polynomial', 0, 10, st.session_state['Degree_SVM'],
                                           key='SVM_min_sample_split')
                    st.session_state['Degree_SVM'] = Degree_SVM

                    st.button('Build Model',key='build_SVM')

                with col_SVM3:
                    Data_model = pd.concat([df[features], df[target]], axis=1)
                    st.write(Data_model)
            with KNN:

                features = st.multiselect(
                    'features', df.columns, key='KNN_Feature'
                )
                colKNN1, colKNN2 = st.columns(2)
                with colKNN2:
                    target = st.selectbox(
                        'Target Value',
                        df.columns, key='KNN_Target')
                with colKNN1:

                    Train_Test_Split_KNN = st.slider('Train Test Split', 0, 100,
                                                     st.session_state['Train_Test_Split_KNN'],
                                                     key='KNN_Train_Test_Split')
                    st.session_state['Train_Test_Split_KNN'] = Train_Test_Split_KNN
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

                    st.button('Build Model', key='build_KNN')

                with col_KNN3:
                    Data_model = pd.concat([df[features], df[target]], axis=1)
                    st.write(Data_model)

if __name__ == '__main__':
    main()
