import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

Future main() async {
  final data = await fromCsv('lib/_datasets/advertising.csv',
      columns: [1, 2, 3, 4]);

  final features = data.toMatrix().submatrix(columns: ZRange.closed(0, 2));
  final labels = data.toMatrix().submatrix(columns: ZRange.singleton(3));

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
