import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future knnRegression() async {
  final data = await fromCsv('lib/_datasets/housing.csv',
    headerExists: false,
    columnDelimiter: ' ',
  );

  final normalized = Normalizer().process(data);

  final folds = 5;
  final validator = CrossValidator.kFold(
    normalized,
    ['col_13'],
    numberOfFolds: folds,
    dtype: DType.float64,
  );

  final scores =
    await validator.evaluate((trainSamples, targetNames) =>
        KnnRegressor(trainSamples, targetNames.first, folds), MetricType.mape);

  print('Boston housing dataset, KNN regression, MAPE error on k-fold '
      'validation ($folds folds): ${scores.mean().toStringAsFixed(2)}%');
}
