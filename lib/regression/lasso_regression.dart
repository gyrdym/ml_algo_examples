import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future main() async {
  final data = DataFrame.fromCsv('lib/_datasets/advertising.csv',
      columns: [ZRange.closed(1, 4)], labelName: 'Sales');
  final features = await data.features;
  final labels = await data.labels;
  final validator = CrossValidator.kFold(numberOfFolds: 5);
  final error = validator.evaluate((Matrix features, Matrix outcomes) =>
      LinearRegressor.coordinate(
          features,
          outcomes,
          iterationsLimit: 100,
          lambda: 46420.0),
      features, labels, MetricType.mape);

  print('error: ${error.toStringAsFixed(2)}%');
}
