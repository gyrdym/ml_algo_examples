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

  final logisticRegressorFactory = (trainSamples, targetNames) =>
      LogisticRegressor(
        trainSamples,
        targetNames.first,
        optimizerType: LinearOptimizerType.gradient,
        initialLearningRate: 0.01,
        iterationsLimit: 100,
        minCoefficientsUpdate: null,
        fitIntercept: true,
        interceptScale: 1,
        batchSize: trainSamples.rows.length,
        learningRateType: LearningRateType.constant,
      );

  final onDataSplit = (trainData, testData) {
    final pipeline = Pipeline(trainData, [
      standardize(),
    ]);

    final trainDataTransformed = pipeline.process(trainData);
    final testDataTransformed = pipeline.process(testData);

    return [
      trainDataTransformed,
      testDataTransformed,
    ];
  };

  final accuracy = validator.evaluate(
    logisticRegressorFactory,
    MetricType.accuracy,
    onDataSplit: onDataSplit,
  );

  print('Pima indians diabetes dataset, binary classification: accuracy is '
      '${accuracy.toStringAsFixed(3)}');
}

Future main() async {
  await binaryClassification();
}
