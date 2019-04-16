import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future main() async {
  final data = DataFrame.fromCsv('lib/_datasets/advertising.csv',
      columns: [ZRange.closed(1, 4)], labelName: 'Sales');
  final features = await data.features;
  final labels = await data.labels;
  final model = LinearRegressor.lasso(iterationsLimit: 100, lambda: 46420.0);
  final validator = CrossValidator.kFold();
  final error = validator.evaluate(model, features, labels, MetricType.mape);

  print('coefficients: ${model.weights}');
  print('error: ${error.toStringAsFixed(2)}%');
}
