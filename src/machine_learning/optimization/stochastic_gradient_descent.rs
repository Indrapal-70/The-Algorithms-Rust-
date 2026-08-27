/// Stochastic Gradient Descent (SGD) Optimization
///
/// Stochastic Gradient Descent is an iterative optimization algorithm used to find the minimum
/// of an objective function. Unlike batch gradient descent, which computes the gradient using
/// the entire dataset before making a single update, SGD updates the parameters incrementally
/// after evaluating the gradient for each individual training sample.
///
/// This sample-by-sample update rule allows SGD to make frequent parameter updates, which can lead
/// to faster initial convergence and helps navigate large datasets efficiently.
///
/// The update equation for parameter vector $x$ given a single data sample $d_i$ is:
/// $$x_{k+1} = x_k - \text{learning\_rate} \times \nabla f(x_k, d_i)$$
///
/// # Arguments
///
/// * `sample_derivative_fn` - A function calculating the gradient for a single data sample at parameter vector `x`.
/// * `x` - The initial parameter vector to be optimized (updated in-place).
/// * `data` - A slice of training data samples of type `T`.
/// * `learning_rate` - Step size for each parameter update.
/// * `epochs` - The number of complete passes over the dataset.
///
/// # Returns
///
/// A reference to the optimized parameter vector `x`.
pub fn stochastic_gradient_descent<'a, T>(
    sample_derivative_fn: impl Fn(&[f64], &T) -> Vec<f64>,
    x: &'a mut Vec<f64>,
    data: &[T],
    learning_rate: f64,
    epochs: usize,
) -> &'a mut Vec<f64> {
    for _ in 0..epochs {
        for sample in data {
            let gradient = sample_derivative_fn(x, sample);
            for (x_k, grad) in x.iter_mut().zip(gradient.iter()) {
                *x_k -= learning_rate * grad;
            }
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_convergence_quadratic() {
        // Sample-based quadratic objective: f_i(x) = (x[0] - target_i)^2
        // Gradient for sample target_i: 2 * (x[0] - target_i)
        // Minimum of total objective sum_i (x[0] - target_i)^2 is at mean(targets) = 2.0
        fn sample_derivative(params: &[f64], target: &f64) -> Vec<f64> {
            vec![2.0 * (params[0] - target)]
        }

        let targets = vec![1.0, 2.0, 3.0];
        let mut x = vec![10.0];
        let learning_rate = 0.005;
        let epochs = 1000;

        let minimized =
            stochastic_gradient_descent(sample_derivative, &mut x, &targets, learning_rate, epochs);

        let expected_min = 2.0;
        let tolerance = 0.02;
        assert!((minimized[0] - expected_min).abs() < tolerance);
    }

    #[test]
    fn test_sgd_unoptimized() {
        fn sample_derivative(params: &[f64], target: &f64) -> Vec<f64> {
            vec![2.0 * (params[0] - target)]
        }

        let targets = vec![1.0, 2.0, 3.0];
        let mut x = vec![10.0];
        let learning_rate = 0.005;
        let epochs = 1;

        let minimized =
            stochastic_gradient_descent(sample_derivative, &mut x, &targets, learning_rate, epochs);

        let expected_min = 2.0;
        let tolerance = 0.02;
        assert!((minimized[0] - expected_min).abs() >= tolerance);
    }

    #[test]
    fn test_sgd_empty_data() {
        fn sample_derivative(_params: &[f64], _sample: &f64) -> Vec<f64> {
            vec![1.0]
        }

        let mut x = vec![5.0, 6.0];
        let initial_x = x.clone();
        let empty_data: Vec<f64> = vec![];

        let minimized =
            stochastic_gradient_descent(sample_derivative, &mut x, &empty_data, 0.01, 100);

        assert_eq!(minimized, &initial_x);
    }

    #[test]
    fn test_sgd_empty_params() {
        fn sample_derivative(_params: &[f64], _sample: &f64) -> Vec<f64> {
            vec![]
        }

        let mut x: Vec<f64> = vec![];
        let data = vec![1.0, 2.0];

        let minimized = stochastic_gradient_descent(sample_derivative, &mut x, &data, 0.01, 100);

        assert!(minimized.is_empty());
    }

    #[test]
    fn test_sgd_single_sample() {
        fn sample_derivative(params: &[f64], target: &f64) -> Vec<f64> {
            vec![2.0 * (params[0] - target)]
        }

        let data = vec![5.0];
        let mut x = vec![0.0];
        let learning_rate = 0.05;
        let epochs = 200;

        let minimized =
            stochastic_gradient_descent(sample_derivative, &mut x, &data, learning_rate, epochs);

        let tolerance = 1e-4;
        assert!((minimized[0] - 5.0).abs() < tolerance);
    }

    #[test]
    fn test_sgd_linear_regression() {
        // Fits y = 2.0 * x + 1.0 using sample-by-sample updates
        // params: [slope, intercept]
        fn sample_derivative(params: &[f64], sample: &(f64, f64)) -> Vec<f64> {
            let (x_val, y_val) = *sample;
            let pred = params[0] * x_val + params[1];
            let error = pred - y_val;
            vec![2.0 * error * x_val, 2.0 * error]
        }

        let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
        let mut params = vec![0.0, 0.0];
        let learning_rate = 0.02;
        let epochs = 1000;

        let minimized = stochastic_gradient_descent(
            sample_derivative,
            &mut params,
            &data,
            learning_rate,
            epochs,
        );

        let tolerance = 1e-3;
        assert!((minimized[0] - 2.0).abs() < tolerance);
        assert!((minimized[1] - 1.0).abs() < tolerance);
    }
}
