import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
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
  );

  final error =
    validator.evaluate((trainSamples, targetNames) =>
        KnnRegressor(
            trainSamples,
            targetNames.first,
            k: 4,
            distance: Distance.euclidean
        ), MetricType.mape);

  print('KNN regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}
