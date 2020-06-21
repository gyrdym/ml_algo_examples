import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future decisionTreeClassification() async {
  final samples = (await fromCsv('lib/_datasets/iris.csv'))
      .dropSeries(seriesNames: ['Id']);
  final pipeline = Pipeline(samples, [
    encodeAsIntegerLabels(
      featureNames: ['Species'],
    ),
  ]);
  final numberOfFolds = 5;
  final processed = pipeline.process(samples);
  final validator = CrossValidator.kFold(
    processed,
    ['Species'],
    numberOfFolds: numberOfFolds,
  );
  final createClassifier = (trainSamples, targetNames) =>
      DecisionTreeClassifier(
        trainSamples,
        targetNames.first,
        minError: 0.2,
        minSamplesCount: 3,
        maxDepth: 7,
      );
  final scores = await validator.evaluate(
    createClassifier,
    MetricType.accuracy,
  );

  print('Iris dataset, decision tree classifier: accuracy on k-fold validation '
      '($numberOfFolds folds): ${scores.mean().toStringAsFixed(2)}');
}

void main() async {
  await decisionTreeClassification();
}
