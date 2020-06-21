import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

Future logisticRegression() async {
  final samples = await fromCsv('lib/_datasets/pima_indians_diabetes_database.csv');
  final numberOfFolds = 5;
  final validator = CrossValidator.kFold(
    samples,
    ['class variable (0 or 1)'],
    numberOfFolds: numberOfFolds,
  );
  final logisticRegressorFactory = (trainSamples, targetNames) =>
    LogisticRegressor(
      trainSamples,
      targetNames.first,
      learningRateType: LearningRateType.decreasingAdaptive,
      probabilityThreshold: 0.7,
      fitIntercept: true,
      interceptScale: .1,
      randomSeed: 3,
    );
  final scores = await validator.evaluate(
    logisticRegressorFactory,
    MetricType.accuracy,
  );

  print('Pima indians diabetes dataset, logistic regression: accuracy on '
      'k-fold validation ($numberOfFolds folds): '
      '${scores.mean().toStringAsFixed(2)}');
}

Future main() async {
  await logisticRegression();
}
