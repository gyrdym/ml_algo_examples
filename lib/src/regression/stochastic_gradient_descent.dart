import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future sgdRegression() async {
  final samples = await fromCsv('lib/_datasets/housing.csv',
      headerExists: false,
      columnDelimiter: ' ');

//  final normalized = Normalizer().process(samples);

  final folds = 5;

  final validator = CrossValidator.kFold(
    samples,
    ['col_13'],
    numberOfFolds: folds,
  );

  final error =
    validator.evaluate((trainSamples, targetNames) =>
        LinearRegressor(
            trainSamples,
            targetNames.first,
            optimizerType: LinearOptimizerType.gradient,
            iterationsLimit: 1000,
            initialLearningRate: 0.05,
            minCoefficientsUpdate: null,
            learningRateType: LearningRateType.decreasingAdaptive,
            dtype: DType.float64),
        MetricType.mape);

  print('SGD regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}

void main() async {
  await sgdRegression();
}
