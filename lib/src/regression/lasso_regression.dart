import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

Future lassoRegression() async {
  final samples = (await fromCsv('lib/_datasets/advertising.csv'))
      .dropSeries(seriesNames: ['Num']);

  final validator = CrossValidator.kFold(
    samples,
    ['Sales'],
    numberOfFolds: 5,
  );

  final scores = await validator.evaluate((trainSamples, targetNames) =>
      LinearRegressor(
          trainSamples,
          targetNames.first,
          optimizerType: LinearOptimizerType.coordinate,
          iterationsLimit: 100,
          lambda: 46420.0),
      MetricType.mape);

  print('Lasso regression, error: ${scores.mean().toStringAsFixed(2)}%');
}
