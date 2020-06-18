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
  final error =
    validator.evaluate((trainSamples, targetNames) =>
        LinearRegressor(
            trainSamples,
            targetNames.first,
            optimizerType: LinearOptimizerType.gradient,
            initialLearningRate: 0.00000385,
            randomSeed: 2,
            learningRateType: LearningRateType.decreasingAdaptive),
        MetricType.mape);

  print('SGD regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}

void main() async {
  await sgdRegression();
}
