import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.colors import n_colors


class autoViz:
    """This class implements model visualization.
    
    Parameters
    ----------
    preprocess_dict : dict, default = None
        1st output result (DICT_PREPROCESS) of autoPipe module.

    report : df, default = None
        4th output result (dyna_report) of autoPipe module.

    Example
    -------
    
    .. [Example]: 
    
    References
    ----------
    
    """
    def __init__(self,preprocess_dict = None,report = None ):
        self.DICT_PREPROCESSING = preprocess_dict
        self.dyna_report = report

    def clf_table_report(self):
        """This function implements heatmap style pipeline cluster's model evaluation report(Dynamic Table) for classification output report.
    
        Parameters
        ----------

        Example
        -------
        
        .. [Example] https://optimal-flow.readthedocs.io/en/latest/demos.html#pipeline-cluster-model-evaluation-dynamic-table-using-autoviz
        
        References
        ----------
        
        """
        df = self.dyna_report
        colors = n_colors('rgb(49, 130, 189)', 'rgb(239, 243, 255)', 15, colortype='rgb')
        bins = [-1, 2, 4, 6, 7, 8, 9, 11]
        bins_latency = [0, 5, 10, 15, 20, 50, 80, 100]
        labels = [1,2,3,4,5,6,7]
        

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        font=dict(color='black', size=12),
                        align='center'),
            cells=dict(values=[df.Dataset,df.Model_Name,df.Best_Parameters,df.Accuracy,df.Precision,df.Recall,df.Latency],
                    # fill_color='lavender',
                    fill_color=['lavender','lavender','lavender',
                    np.array(colors)[pd.cut(df.Accuracy.apply(lambda x: x*10), bins=bins, labels=labels).astype(int)],
                    np.array(colors)[pd.cut(df.Precision.apply(lambda x: x*10), bins=bins, labels=labels).astype(int)],
                    np.array(colors)[pd.cut(df.Recall.apply(lambda x: x*10), bins=bins, labels=labels).astype(int)],
                    'lavender'],
                    align='left'))
                    # np.array(colors)[pd.cut(df.Latency,bins=bins_latency, labels=labels).astype(int)]],        
        ])
        fig.update_layout(title = f'Pipeline Cluster Model Classification Evaluation Report - autoViz <a href="https://www.linkedin.com/in/lei-tony-dong/"> ©Tony Dong</a>', font_size=8)
        plot(fig)
        fig.show()


    def reg_table_report(self):
        """This function implements heatmap style pipeline cluster's model evaluation report(Dynamic Table) for regression output report.
    
        Parameters
        ----------

        Example
        -------
        
        .. [Example] https://optimal-flow.readthedocs.io/en/latest/demos.html#pipeline-cluster-model-evaluation-dynamic-table-using-autoviz
        
        References
        ----------
        
        """
        df = self.dyna_report
        colors = n_colors('rgb(49, 130, 189)', 'rgb(239, 243, 255)', 15, colortype='rgb')
        bins = [-1, 2, 4, 6, 7, 8, 9, 11]
        labels = [1,2,3,4,5,6,7]      

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
            cells=dict(values=[df.Dataset,df.Model_Name,df.Best_Parameters,df.R2,df.MAE,df.MSE,df.RMSE,df.Latency],
                    # fill_color='lavender',
                    fill_color=['lavender','lavender','lavender',
                    np.array(colors)[pd.cut(df.R2.apply(lambda x: x*10), bins=bins, labels=labels).astype(int)],
                    'lavender',
                    'lavender',
                    'lavender',
                    'lavender'],
                    align='left'))
        ])
        fig.update_layout(title = f'Pipeline Cluster Model Regression Evaluation Report - autoViz <a href="https://www.linkedin.com/in/lei-tony-dong/"> ©Tony Dong</a>', font_size=8)
        plot(fig)
        fig.show()

    def clf_model_retrieval(self,metrics = None):
        """This function implements classification model retrieval visualization.
    
        Parameters
        ----------
        metrics : str, default = None
            Value in ["accuracy","precision","recall"].

        Example
        -------
        
        .. [Example] https://optimal-flow.readthedocs.io/en/latest/demos.html#pipeline-cluster-traversal-experiments-model-retrieval-diagram-using-autoviz
        
        References
        ----------
        
        """
        columns = ["Dataset","Encode_low_dimension","Encode_high_dimension","Winsorize","Scale"]
        df_pp = pd.DataFrame(columns=columns)

        for i in list(self.DICT_PREPROCESSING.keys()):
            row_pp = [i]
            s = self.DICT_PREPROCESSING[i]
            ext = re.search("Encoded Features:(.*)']", s).group(1)
            
            if ("onehot_" in ext) and ("Frequency_" in ext):
                row_pp.append('Low Dim_Onehot')
                row_pp.append('High Dim_Frequency')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp
            elif ("onehot_" in ext) and ("Mean_" in ext):
                row_pp.append('Low Dim_Onehot')
                row_pp.append('High Dim_Mean')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp
            elif ("onehot_" in ext) and ("Mean_" not in ext) and ("Frequency_" not in ext):
                row_pp.append('Low Dim_Onehot')
                row_pp.append('High Dim_No Encoder')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp
            elif ("Label_" in ext) and ("Frequency_" in ext):
                row_pp.append('Low Dim_Label')
                row_pp.append('High Dim_Frequency')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp
            elif ("Label_" in ext) and ("Mean_" in ext):
                row_pp.append('Low Dim_Label')
                row_pp.append('High Dim_Mean')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp
            elif ("Label_" in ext) and ("Mean_" not in ext) and ("Frequency_" not in ext):
                row_pp.append('Low Dim_Label')
                row_pp.append('High Dim_No Encoder')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp
            elif ("Frequency_" in ext) and ("onehot_" not in ext) and ("Label_" not in ext):
                row_pp.append('Low Dim_No Encoder')
                row_pp.append('High Dim_Frequency')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp    
            elif ("Mean_" in ext) and ("onehot_" not in ext) and ("Label_" not in ext):
                row_pp.append('Low Dim_No Encoder')
                row_pp.append('High Dim_Mean')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp    
            elif ("Frequency_" not in ext) and ("Mean_" not in ext) and ("onehot_" not in ext) and ("Label_" not in ext):
                row_pp.append('Low Dim_No Encoder')
                row_pp.append('High Dim_No Encoder')
                row_pp.append(re.search('winsor_(.*)-Scaler', s).group(1))
                row_pp.append(re.search('-Scaler_(.*)-- ', s).group(1))
                df_pp.loc[len(df_pp)] = row_pp 

        if metrics == "accuracy":
            df_report_Accuracy = df_pp.merge(self.dyna_report[['Dataset','Accuracy']], how = 'left', on = 'Dataset')
            bins = [0, 0.70, 0.90, 1]
            labels = ["Low Accuracy","High Accuracy","Top Accuracy"]
            df_report_Accuracy['Level'] = pd.cut(df_report_Accuracy['Accuracy'], bins=bins, labels=labels)
            df_report_Accuracy['cnt'] = 1
            df_report_Accuracy.loc[df_report_Accuracy['Scale'] == 'None','Scale'] = "No Scaler"
            df_report_Accuracy['Scale'] = 'Scale_'+df_report_Accuracy['Scale']
            df_report_Accuracy['Winsorize'] = 'Winsorize_' + df_report_Accuracy['Winsorize']

            step1_df = df_report_Accuracy.groupby(['Encode_low_dimension','Dataset'], as_index=False)['cnt'].count().rename({"cnt":"Total","Dataset":"antecedentIndex","Encode_low_dimension":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step2_df = df_report_Accuracy.groupby(['Encode_low_dimension','Encode_high_dimension'], as_index=False)['cnt'].count().rename({"cnt":"Total","Encode_low_dimension":"antecedentIndex","Encode_high_dimension":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step3_df = df_report_Accuracy.groupby(['Encode_high_dimension','Winsorize'], as_index=False)['cnt'].count().rename({"cnt":"Total","Encode_high_dimension":"antecedentIndex","Winsorize":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step4_df = df_report_Accuracy.groupby(['Winsorize','Scale'], as_index=False)['cnt'].count().rename({"cnt":"Total","Winsorize":"antecedentIndex","Scale":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step5_df = df_report_Accuracy.groupby(['Scale','Level'], as_index=False)['cnt'].count().rename({"cnt":"Total","Scale":"antecedentIndex","Level":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']].dropna()
            integrated_df = pd.concat([step1_df,step2_df,step3_df,step4_df,step5_df],axis = 0)

            label_df = pd.DataFrame(integrated_df['antecedentIndex'].append(integrated_df['consequentIndex']).drop_duplicates(),columns = {"label"})
            label_df['Number'] = label_df.reset_index().index
            label_list = list(label_df.label)

            source_df = pd.DataFrame(integrated_df['antecedentIndex'])
            source_df = source_df.merge(label_df, left_on=['antecedentIndex'], right_on = ['label'],how = 'left')
            source_list = list(source_df['Number'])

            target_df = pd.DataFrame(integrated_df['consequentIndex'])
            target_df = target_df.merge(label_df, left_on=['consequentIndex'], right_on = ['label'],how = 'left')
            target_list = list(target_df['Number'])

            value_list = [int(i) for i in list(integrated_df.Total)]

            fig = go.Figure(data=[go.Sankey(
                node = dict(
                pad = 15,
                thickness = 10,
                line = dict(color = 'rgb(25,100,90)', width = 0.5),
                label = label_list,
                color = 'rgb(71,172,55)'
                ),
                link = dict(
                source = source_list, 
                target = target_list,
                value = value_list
            ))])

            fig.update_layout(title = f'Pipeline Cluster Traversal Experiments - autoViz {metrics} Retrieval Diagram <a href="https://www.linkedin.com/in/lei-tony-dong/"> ©Tony Dong</a>', font_size=8)
            plot(fig)
            fig.show()

        elif metrics == "precision":
            df_report_Precision = df_pp.merge(self.dyna_report[['Dataset','Precision']], how = 'left', on = 'Dataset')
            bins = [0, 0.70, 0.90, 1]
            labels = ["Low Precision","High Precision","Top Precision"]
            df_report_Precision['Level'] = pd.cut(df_report_Precision['Precision'], bins=bins, labels=labels)
            df_report_Precision['cnt'] = 1
            df_report_Precision.loc[df_report_Precision['Scale'] == 'None','Scale'] = "No Scaler"
            df_report_Precision['Scale'] = 'Scale_'+df_report_Precision['Scale']
            df_report_Precision['Winsorize'] = 'Winsorize_' + df_report_Precision['Winsorize']
            
            step1_df = df_report_Precision.groupby(['Encode_low_dimension','Dataset'], as_index=False)['cnt'].count().rename({"cnt":"Total","Dataset":"antecedentIndex","Encode_low_dimension":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step2_df = df_report_Precision.groupby(['Encode_low_dimension','Encode_high_dimension'], as_index=False)['cnt'].count().rename({"cnt":"Total","Encode_low_dimension":"antecedentIndex","Encode_high_dimension":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step3_df = df_report_Precision.groupby(['Encode_high_dimension','Winsorize'], as_index=False)['cnt'].count().rename({"cnt":"Total","Encode_high_dimension":"antecedentIndex","Winsorize":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step4_df = df_report_Precision.groupby(['Winsorize','Scale'], as_index=False)['cnt'].count().rename({"cnt":"Total","Winsorize":"antecedentIndex","Scale":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step5_df = df_report_Precision.groupby(['Scale','Level'], as_index=False)['cnt'].count().rename({"cnt":"Total","Scale":"antecedentIndex","Level":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']].dropna()
            integrated_df = pd.concat([step1_df,step2_df,step3_df,step4_df,step5_df],axis = 0)

            label_df = pd.DataFrame(integrated_df['antecedentIndex'].append(integrated_df['consequentIndex']).drop_duplicates(),columns = {"label"})
            label_df['Number'] = label_df.reset_index().index
            label_list = list(label_df.label)

            source_df = pd.DataFrame(integrated_df['antecedentIndex'])
            source_df = source_df.merge(label_df, left_on=['antecedentIndex'], right_on = ['label'],how = 'left')
            source_list = list(source_df['Number'])

            target_df = pd.DataFrame(integrated_df['consequentIndex'])
            target_df = target_df.merge(label_df, left_on=['consequentIndex'], right_on = ['label'],how = 'left')
            target_list = list(target_df['Number'])

            value_list = [int(i) for i in list(integrated_df.Total)]

            fig = go.Figure(data=[go.Sankey(
                node = dict(
                pad = 15,
                thickness = 10,
                line = dict(color = 'rgb(25,100,90)', width = 0.5),
                label = label_list,
                color = 'rgb(71,172,55)'
                ),
                link = dict(
                source = source_list, 
                target = target_list,
                value = value_list
            ))])

            fig.update_layout(title = f'Pipeline Cluster Traversal Experiments - autoViz {metrics} Retrieval Diagram <a href="https://www.linkedin.com/in/lei-tony-dong/"> ©Tony Dong</a>', font_size=8)
            plot(fig)
            fig.show()

        elif metrics == "recall":
            df_report_Recall = df_pp.merge(dyna_report[['Dataset','Recall']], how = 'left', on = 'Dataset')
            bins = [0, 0.70, 0.90, 1]
            labels = ["Low Recall","High Recall","Top Recall"]
            df_report_Recall['Level'] = pd.cut(df_report_Recall['Recall'], bins=bins, labels=labels)
            df_report_Recall['cnt'] = 1
            df_report_Recall.loc[df_report_Recall['Scale'] == 'None','Scale'] = "No Scaler"
            df_report_Recall['Scale'] = 'Scale_'+df_report_Recall['Scale']
            df_report_Recall['Winsorize'] = 'Winsorize_' + df_report_Recall['Winsorize']

            step1_df = df_report_Recall.groupby(['Encode_low_dimension','Dataset'], as_index=False)['cnt'].count().rename({"cnt":"Total","Dataset":"antecedentIndex","Encode_low_dimension":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step2_df = df_report_Recall.groupby(['Encode_low_dimension','Encode_high_dimension'], as_index=False)['cnt'].count().rename({"cnt":"Total","Encode_low_dimension":"antecedentIndex","Encode_high_dimension":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step3_df = df_report_Recall.groupby(['Encode_high_dimension','Winsorize'], as_index=False)['cnt'].count().rename({"cnt":"Total","Encode_high_dimension":"antecedentIndex","Winsorize":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step4_df = df_report_Recall.groupby(['Winsorize','Scale'], as_index=False)['cnt'].count().rename({"cnt":"Total","Winsorize":"antecedentIndex","Scale":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']]
            step5_df = df_report_Recall.groupby(['Scale','Level'], as_index=False)['cnt'].count().rename({"cnt":"Total","Scale":"antecedentIndex","Level":"consequentIndex"},axis = 1)[['antecedentIndex','consequentIndex','Total']].dropna()
            integrated_df = pd.concat([step1_df,step2_df,step3_df,step4_df,step5_df],axis = 0)

            label_df = pd.DataFrame(integrated_df['antecedentIndex'].append(integrated_df['consequentIndex']).drop_duplicates(),columns = {"label"})
            label_df['Number'] = label_df.reset_index().index
            label_list = list(label_df.label)

            source_df = pd.DataFrame(integrated_df['antecedentIndex'])
            source_df = source_df.merge(label_df, left_on=['antecedentIndex'], right_on = ['label'],how = 'left')
            source_list = list(source_df['Number'])

            target_df = pd.DataFrame(integrated_df['consequentIndex'])
            target_df = target_df.merge(label_df, left_on=['consequentIndex'], right_on = ['label'],how = 'left')
            target_list = list(target_df['Number'])

            value_list = [int(i) for i in list(integrated_df.Total)]

            fig = go.Figure(data=[go.Sankey(
                node = dict(
                pad = 15,
                thickness = 10,
                line = dict(color = 'rgb(25,100,90)', width = 0.5),
                label = label_list,
                color = 'rgb(71,172,55)'
                ),
                link = dict(
                source = source_list, 
                target = target_list,
                value = value_list
            ))])

            fig.update_layout(title = f'Pipeline Cluster Traversal Experiments - autoViz {metrics} Retrieval Diagram <a href="https://www.linkedin.com/in/lei-tony-dong/"> ©Tony Dong</a>', font_size=8)
            plot(fig)
            fig.show()



