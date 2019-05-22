import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future binaryClassification() async {
  final data = DataFrame.fromCsv('lib/_datasets/pima_indians_diabetes_database.csv',
    labelName: 'class variable (0 or 1)',
  );

  final features = (await data.features)
      .mapColumns((column) => column.normalize());
  final labels = await data.labels;
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
