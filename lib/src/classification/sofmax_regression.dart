import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future softmaxRegression() async {
  final data = (await fromCsv('lib/_datasets/iris.csv'))
      .dropSeries(seriesNames: ['Id']);

  final pipeline = Pipeline(data, [
    encodeAsOneHotLabels(featureNames: ['Species']),
    normalize(),
  ]);

  final processed = pipeline
      .process(data)
      .toMatrix();

  final features = processed.submatrix(columns: ZRange.closed(1, 4));
  final labels = processed.submatrix(columns: ZRange.singleton(5));

  final validator = CrossValidator.kFold(numberOfFolds: 5);
  final accuracy = validator.evaluate(
          (Matrix features, Matrix labels) =>
              SoftmaxRegressor.gradient(
                features,
                labels,
                initialLearningRate: 0.03,
                iterationsLimit: 200,
                minWeightsUpdate: 1e-6,
                randomSeed: 46,
                learningRateType: LearningRateType.constant),
      features, labels, MetricType.accuracy);

  print('Iris dataset, softmax regression: accuracy is '
      '${accuracy.toStringAsFixed(2)}');
}

Future main() async {
  await softmaxRegression();
}
