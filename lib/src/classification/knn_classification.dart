import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future knnClassification() async {
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
    numberOfFolds: 7,
  );

  final accuracy = validator.evaluate((trainSamples, targetNames) =>
      KnnClassifier(
        trainSamples,
        'Species',
        5,
      ),
    MetricType.accuracy,
  );

  print('Iris dataset, KNN classifier, accuracy is '
      '${accuracy.toStringAsFixed(2)}');
}

void main() async {
  await knnClassification();
}
