import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future main() async {
  final data = await fromCsv('lib/_datasets/housing.csv',
    headerExists: false,
    columnDelimiter: ' ',
  );

  final normalized = Normalizer().process(data);

  final features = normalized
      .toMatrix()
      .submatrix(columns: ZRange.closed(0, 12));

  final labels = normalized
      .toMatrix()
      .submatrix(columns: ZRange.singleton(13));

  final folds = 5;
  final validator = CrossValidator.kFold(numberOfFolds: folds);

  final error =
    validator.evaluate((Matrix features, Matrix outcomes) =>
        ParameterlessRegressor.knn(
            features,
            outcomes,
            k: 4,
            distance: Distance.euclidean
        ), features, labels, MetricType.mape);

  print('KNN regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}
