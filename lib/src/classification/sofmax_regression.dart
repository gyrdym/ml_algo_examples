import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future softmaxRegression() async {
  final samples = (await fromCsv('lib/_datasets/iris.csv'))
      .dropSeries(seriesNames: ['Id']);

  final pipeline = Pipeline(samples, [
    encodeAsOneHotLabels(
        featureNames: ['Species'],
    ),
    normalize(),
  ]);

  final processed = pipeline.process(samples);

  final validator = CrossValidator.kFold(
    processed,
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    numberOfFolds: 5,
  );

  final predictorFactory = (trainSamples, _) =>
      SoftmaxRegressor(
        trainSamples,
        ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
        initialLearningRate: 0.03,
        iterationsLimit: 200,
        minCoefficientsUpdate: 1e-6,
        randomSeed: 46,
        learningRateType: LearningRateType.constant,
      );

  final onDataSplit = (trainData, testData) {
    final standardizer = Standardizer(trainData);
    return [
      standardizer.process(trainData),
      standardizer.process(testData),
    ];
  };

  final accuracy = validator.evaluate(
    predictorFactory,
    MetricType.accuracy,
    onDataSplit: onDataSplit,
  );

  print('Iris dataset, softmax regression: accuracy is '
      '${accuracy.toStringAsFixed(2)}');
}

Future main() async {
  await softmaxRegression();
}
