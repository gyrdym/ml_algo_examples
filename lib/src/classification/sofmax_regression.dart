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
  ]);
  final processed = pipeline.process(samples);
  final numberOfFolds = 5;
  final validator = CrossValidator.kFold(
    processed,
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    numberOfFolds: numberOfFolds,
  );
  final predictorFactory = (trainSamples, _) =>
      SoftmaxRegressor(
        trainSamples,
        ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
        initialLearningRate: 0.035,
        iterationsLimit: 5000,
        minCoefficientsUpdate: null,
        learningRateType: LearningRateType.constant,
      );
  final scores = await validator.evaluate(
    predictorFactory,
    MetricType.accuracy,
  );

  print('Iris dataset, softmax regression: accuracy on k-fold validation '
      '($numberOfFolds folds): ${scores.mean().toStringAsFixed(2)}');
}

Future main() async {
  await softmaxRegression();
}
