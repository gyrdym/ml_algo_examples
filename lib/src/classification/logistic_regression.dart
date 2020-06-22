import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

Future logisticRegression() async {
  final samples = await fromCsv('lib/_datasets/pima_indians_diabetes_database.csv');
  final splits = splitData(samples, [0.7]);
  final validationData = splits[0];
  final testData = splits[1];
  final numberOfFolds = 5;
  final targetNames = ['class variable (0 or 1)'];
  final validator = CrossValidator.kFold(
    validationData,
    targetNames,
    numberOfFolds: numberOfFolds,
  );
  final createClassifier = (DataFrame trainSamples, targetNames) =>
    LogisticRegressor(
      trainSamples,
      targetNames.first,
      fitIntercept: true,
      interceptScale: 200,
      learningRateType: LearningRateType.decreasingAdaptive,
      batchSize: trainSamples.rows.length,
    );
  final scores = await validator.evaluate(
    createClassifier,
    MetricType.accuracy,
  );

  print('Pima indians diabetes dataset, Logistic regression');

  print('Accuracy on k-fold validation ($numberOfFolds folds): '
      '${scores.mean().toStringAsFixed(2)}');

  final testSplits = splitData(testData, [0.8]);
  final classifier = createClassifier(testSplits[0], targetNames);
  final finalScore = classifier.assess(testSplits[1], targetNames,
      MetricType.accuracy);

  print('Accuracy on the model assessment: ${finalScore.toStringAsFixed(2)}');
}

Future main() async {
  await logisticRegression();
}
