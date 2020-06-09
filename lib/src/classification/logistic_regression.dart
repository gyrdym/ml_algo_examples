import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

Future logisticRegression() async {
  final samples = await fromCsv('lib/_datasets/pima_indians_diabetes_database.csv');

  final validator = CrossValidator.kFold(
    samples,
    ['class variable (0 or 1)'],
    numberOfFolds: 5,
  );

  final logisticRegressorFactory = (trainSamples, targetNames) {
    final classifier = LogisticRegressor(
      trainSamples,
      targetNames.first,
      optimizerType: LinearOptimizerType.gradient,
      learningRateType: LearningRateType.decreasingAdaptive,
      probabilityThreshold: 0.7,
      collectLearningData: true,
    );

    return classifier;
  };

  final accuracy = validator.evaluate(
    logisticRegressorFactory,
    MetricType.accuracy,
  );

  print('Pima indians diabetes dataset, binary classification: accuracy is '
      '${accuracy.toStringAsFixed(5)}');
}

Future main() async {
  await logisticRegression();
}
