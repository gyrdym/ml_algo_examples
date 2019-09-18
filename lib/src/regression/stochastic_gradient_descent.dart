import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future bostonHousingRegression() async {
  final data = await fromCsv('lib/_datasets/housing.csv',
      headerExists: false,
      columnDelimiter: ' ');

  final normalized = Normalizer().process(data);
  final asMatrix = normalized.toMatrix();

  final features = asMatrix.submatrix(columns: ZRange.closed(0, 12));
  final labels = asMatrix.submatrix(columns: ZRange.singleton(13));

  final folds = 5;
  final validator = CrossValidator.kFold(numberOfFolds: folds);

  final error =
    validator.evaluate((Matrix features, Matrix outcomes) =>
        LinearRegressor.gradient(
            features,
            outcomes,
            iterationsLimit: 100,
            initialLearningRate: 5.0,
            minWeightsUpdate: 1e-4,
            randomSeed: 20,
            learningRateType: LearningRateType.constant),
        features, labels, MetricType.mape);

  print('Linear regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}

Future main() async {
  await bostonHousingRegression();
}
