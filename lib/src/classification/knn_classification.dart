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
  final numberOfFolds = 7;
  final numberOfNeighbours = 5;
  final validator = CrossValidator.kFold(
    processed,
    ['Species'],
    numberOfFolds: numberOfFolds,
  );
  final createClassifier = (trainSamples, targetNames) =>
      KnnClassifier(
        trainSamples,
        'Species',
        numberOfNeighbours,
      );
  final scores = await validator.evaluate(
    createClassifier,
    MetricType.accuracy,
  );

  print('Iris dataset, KNN classification: accuracy on k-fold validation '
      '($numberOfFolds folds): ${scores.mean().toStringAsFixed(2)}');
}

void main() async {
  await knnClassification();
}
