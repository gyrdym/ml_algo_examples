import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

Future sgdRegression() async {
  final samples = await fromCsv('lib/_datasets/housing.csv',
      headerExists: false,
      columnDelimiter: ' ');
  final folds = 5;
  final validator = CrossValidator.kFold(
    samples,
    ['col_13'],
    numberOfFolds: folds,
  );
  final createRegressor = (DataFrame trainSamples, targetNames) =>
      LinearRegressor(
          trainSamples,
          targetNames.first,
          optimizerType: LinearOptimizerType.gradient,
          initialLearningRate: 0.00000385,
          randomSeed: 2,
          learningRateType: LearningRateType.decreasingAdaptive);
  final scores = await validator.evaluate(createRegressor, MetricType.mape);

  print('Boston housing dataset, SGD regression, MAPE error on k-fold '
      'validation ($folds folds): ${scores.mean().toStringAsFixed(2)}%');
}

void main() async {
  await sgdRegression();
}
