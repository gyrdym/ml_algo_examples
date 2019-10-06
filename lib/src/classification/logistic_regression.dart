import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future binaryClassification() async {
  final samples = await fromCsv('lib/_datasets/pima_indians_diabetes_database.csv');
  final normalized = Normalizer().process(samples);

  final validator = CrossValidator.kFold(
    normalized,
    ['class variable (0 or 1)'],
    numberOfFolds: 5,
  );

  final accuracy = validator.evaluate(
          (trainSamples, targetNames) =>
              LogisticRegressor(
                  trainSamples,
                  targetNames.first,
                  optimizerType: LinearOptimizerType.gradient,
                  initialLearningRate: .8,
                  iterationsLimit: 500,
                  fitIntercept: true,
                  interceptScale: 2,
                  batchSize: trainSamples.rows.length,
                  learningRateType: LearningRateType.constant,
              ),
      MetricType.accuracy);

  print('Pima indians diabetes dataset, binary classification: accuracy is '
      '${accuracy.toStringAsFixed(3)}');
}

Future main() async {
  await binaryClassification();
}
