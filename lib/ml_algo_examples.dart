import 'package:ml_algo_examples/src/classification/decision_tree.dart';
import 'package:ml_algo_examples/src/classification/logistic_regression.dart';
import 'package:ml_algo_examples/src/classification/sofmax_regression.dart';
import 'package:ml_algo_examples/src/regression/knn_regression.dart';
import 'package:ml_algo_examples/src/regression/lasso_regression.dart';
import 'package:ml_algo_examples/src/regression/stochastic_gradient_descent.dart';

void main() async {
  await decisionTreeRegression();
  await binaryClassification();
  await softmaxRegression();
  await knnRegression();
  await lassoRegression();
  await sgdRegression();
}
