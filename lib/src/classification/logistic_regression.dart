import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future binaryClassification() async {
  final dataFrame = await fromCsv('lib/_datasets/pima_indians_diabetes_database.csv');

  final normalized = Normalizer().process(dataFrame);
  final normalizedAsMatrix = normalized.toMatrix();

  final features = normalizedAsMatrix.submatrix(columns: ZRange.closed(0, 7));
  final labels = normalizedAsMatrix.submatrix(columns: ZRange.singleton(8));

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final accuracy = validator.evaluate(
      (Matrix features, Matrix outcomes) =>
          LogisticRegressor.gradient(
            features, outcomes,
            initialLearningRate: .8,
            iterationsLimit: 500,
            fitIntercept: true,
            interceptScale: .1,
            batchSize: features.rowsNum,
            learningRateType: LearningRateType.constant),
      features,
      labels,
      MetricType.accuracy);

  print('Pima indians diabetes dataset, binary classification: accuracy is '
      '${accuracy.toStringAsFixed(3)}');
}

Future main() async {
  await binaryClassification();
}
