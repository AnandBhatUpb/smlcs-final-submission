import logging
from smlcs.helper.read_data import ReadData
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class DataVisualize:

    def plot_class_counts(classes, counts, *args):
        try:
            y_pos = np.arange(len(classes))
            f, ax = plt.subplots(figsize=(18, 10))
            plt.bar(y_pos, counts, color=['green', 'black', 'red', 'blue', 'cyan', 'grey'])
            plt.xticks(y_pos, classes)
            # plt.savefig('total_count_second_phase.png')
            plt.show()
        except Exception as e:
            print('Exception occurred in plot_class_counts: %s', str(e))

    def plot_runtime_distribution(runtime, *args):
        try:
            sns.distplot(runtime, hist=False, rug=False)
        except Exception as e:
            print('Exception occurred in plot_runtime_distribution: %s', str(e))

    def plot_heat_map(dataset, *args):
        try:
            f, ax = plt.subplots(figsize=(30, 35))
            corr = dataset.iloc[:, :-10].corr()
            sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                        cmap=sns.diverging_palette(220, 10, as_cmap=True),
                        square=True, ax=ax)
            plt.show()
            #plt.savefig('plots/feature_correlation_heatmap_dropped_features.png')
        except Exception as e:
            print('Exception occurred in plot_heat_map: %s', str(e))

    if __name__ == '__main__':
        try:
            logging.basicConfig(filename='../../logs/data_visualization.log', filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
            logger = logging.getLogger('Data_visualization')

            clf_df, reg_df = ReadData('local', logger).read_dataframe(logger)

            runtime = reg_df.iloc[:, 54].tolist()
            classes = clf_df.iloc[:, 54].tolist()

            print(clf_df.iloc[:, 54].shape)
            print(clf_df['class'].value_counts())

            plot_runtime_distribution(runtime, logger)
            plot_heat_map(clf_df, logger)

            counts = [13843, 2620, 8150, 11053, 2798, 10436]
            classes = ('Correct', 'Incorrect', 'Error', 'Exception', 'Unknown', 'Timeout')
            plot_class_counts(classes, counts, logger)

        except Exception as e:
            logger.error('Failed in the main of data_visualize.py: ' + str(e))



