import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future main() async {
  final data = DataFrame.fromCsv('lib/_datasets/housing.csv',
    headerExists: false,
    fieldDelimiter: ' ',
    labelIdx: 13,
  );

  final features = (await data.features)
      .mapColumns((column) => column.rescale());
  final labels = await data.labels;

  final folds = 5;
  final validator = CrossValidator.kFold(numberOfFolds: folds);

  final regressor = NoNParametricRegressor.nearestNeighbor(k: 6,
      distanceType: Distance.euclidean);

  final error =
    validator.evaluate(regressor, features, labels, MetricType.mape);

  print('KNN regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}
