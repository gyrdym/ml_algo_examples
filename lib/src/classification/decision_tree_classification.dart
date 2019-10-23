import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future decisionTreeRegression() async {
  final samples = (await fromCsv('lib/_datasets/iris.csv'))
      .dropSeries(seriesNames: ['Id']);

  final pipeline = Pipeline(samples, [
    encodeAsIntegerLabels(
      featureNames: ['Species'],
    ),
  ]);

  final processed = pipeline.process(samples);

  final validator = CrossValidator.kFold(
    processed,
    ['Species'],
    numberOfFolds: 5,
  );

  final accuracy = validator.evaluate(
          (trainSamples, targetNames) =>
              DecisionTreeClassifier(
                trainSamples,
                targetNames.first,
                minError: 0.2,
                minSamplesCount: 3,
                maxDepth: 7,
              ),
      MetricType.accuracy,
  );

  print('Iris dataset, decision tree classifier: accuracy is '
      '${accuracy.toStringAsFixed(2)}');
}
