import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';

void logisticRegression() async {
  final samples = await fromCsv('lib/_datasets/pima_indians_diabetes_database.csv');
  final shuffledSamples = samples.shuffle(seed: 3);
  final splits = splitData(shuffledSamples, [0.7]);
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
      optimizerType: LinearOptimizerType.gradient,
      iterationsLimit: 50,
      initialCoefficients: Vector.randomFilled(trainSamples.series.length - 1),
      learningRateType: LearningRateType.decreasingAdaptive,
      batchSize: trainSamples.rows.length,
      probabilityThreshold: 0.7,
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
