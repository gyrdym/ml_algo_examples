import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future bostonHousingRegression() async {
  final data = DataFrame.fromCsv('lib/_datasets/housing.csv',
    headerExists: false,
    fieldDelimiter: ' ',
    labelIdx: 13,
  );

  final features = (await data.features)
      .mapColumns((column) => column.normalize());
  final labels = await data.labels;

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
