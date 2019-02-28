import csv


class WriteToCSV:

    def write_result_to_csv(self, logger, *argv):
        try:
            result_string = []
            for i in range(0, len(argv)):
                result_string.append(str(argv[i]))

            with open('../../results/res_result_'+argv[0]+'.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_string)
            f.close()
            logger.info('Writing result to csv file done')
        except Exception as e:
            logger.error('Failed in write_result_to_csv: :' + str(e))