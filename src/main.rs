fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn approx_sigmoid(x: f64) -> f64 {
    let cuts = [-5.0, -2.5, 2.5, 5.0];

    // Calculate le
    let mut le = vec![0.0];
    for &cut in &cuts {
        le.push(if x <= cut { 1.0 } else { 0.0 });
    }
    le.push(1.0);

    // Calculate select
    let mut select = Vec::new();
    for i in 0..5 {
        select.push(le[i + 1] - le[i]);
    }

    let outputs = vec![
        10f64.powi(-4),
        0.02776 * x + 0.145,
        0.17 * x + 0.5,
        0.02776 * x + 0.85498,
        1.0 - 10f64.powi(-4),
    ];

    // Calculate the sum of selected outputs
    select
        .iter()
        .zip(outputs.iter())
        .map(|(&s, &o)| s * o)
        .sum()
}

fn get_prediction(features: &[f64], weights: &[f64], bias: f64) -> f64 {
    approx_sigmoid(dot_product(features, weights) + bias)
}

fn compute_gradient<const N: usize, const M: usize>(
    features: &[[f64; M]; N], // Each row is a data point
    labels: &[f64],           // Corresponding labels
    weights: &[f64],          // Current weights
    bias: f64,                // Current bias
) -> ([f64; M], f64) {
    // Return gradients for weights and bias
    let m = features.len() as f64;

    let mut weight_gradients = vec![0.0; weights.len()];
    let mut bias_gradient = 0.0;

    for i in 0..features.len() {
        let prediction = get_prediction(&features[i], weights, bias);
        let error = prediction - labels[i];

        // Compute gradients for weights
        for j in 0..weights.len() {
            weight_gradients[j] += features[i][j] * error;
        }

        // Compute gradient for bias
        bias_gradient += error;
    }

    // Average the gradients
    let weights: [f64; M] = weight_gradients
        .iter()
        .map(|g| g / m)
        .collect::<Vec<f64>>()
        .try_into()
        .expect("Array size mismatch");
    (weights, bias_gradient / m)
}

fn update_weights(
    weights: &mut [f64],
    bias: &mut f64,
    gradients: &[f64],
    bias_gradient: f64,
    learning_rate: f64,
) {
    for (w, g) in weights.iter_mut().zip(gradients.iter()) {
        *w -= learning_rate * g;
    }
    *bias -= learning_rate * bias_gradient;
}

fn train<const N: usize, const M: usize>(
    features: &[[f64; M]; N],
    labels: &[f64; N],
    learning_rate: f64,
    epochs: usize,
) -> ([f64; M], f64) {
    // Initialize weights and bias within the training function
    let mut weights = [0.0; M]; // Zero-initialized weights
    let mut bias = 0.0; // Zero-initialized bias

    for _ in 0..epochs {
        let (gradients, bias_gradient) = compute_gradient(features, labels, &weights, bias);
        update_weights(
            &mut weights,
            &mut bias,
            &gradients,
            bias_gradient,
            learning_rate,
        );
    }
    (weights, bias)
}

fn train_multi_class<const N: usize, const M: usize, const C: usize>(
    epochs: usize,
    inputs: &[[f64; M]; N],
    labels: &[[f64; N]; C],
    learning_rate: f64,
) -> [([f64; M], f64); C] {
    let mut result_parameters = [([0.0; M], 0.0); C]; // Initialize the result array
    for i in 0..C {
        let parameters = train::<N, M>(
            inputs,
            &labels[i], // Pass labels for the i-th class
            learning_rate,
            epochs,
        );
        result_parameters[i] = parameters;
    }
    result_parameters
}

fn main() {
    println!("Hello, world!");
}

#[test]
fn test_setosa_versicolor() {
    const M: usize = 4;
    const N: usize = 30;
    // Define the dataset (features and labels)
    let inputs: [[f64; M]; N] = [
        [5.5, 2.6, 4.4, 1.2],
        [5.0, 3.4, 1.6, 0.4],
        [5.2, 4.1, 1.5, 0.1],
        [6.5, 2.8, 4.6, 1.5],
        [5.6, 3.0, 4.5, 1.5],
        [4.6, 3.2, 1.4, 0.2],
        [6.0, 2.2, 4.0, 1.0],
        [5.1, 3.4, 1.5, 0.2],
        [6.6, 2.9, 4.6, 1.3],
        [4.7, 3.2, 1.3, 0.2],
        [6.3, 3.3, 4.7, 1.6],
        [5.2, 3.4, 1.4, 0.2],
        [5.8, 2.6, 4.0, 1.2],
        [6.1, 2.8, 4.0, 1.3],
        [5.7, 2.8, 4.1, 1.3],
        [5.8, 4.0, 1.2, 0.2],
        [4.8, 3.4, 1.9, 0.2],
        [5.6, 3.0, 4.5, 1.5],
        [4.9, 3.0, 1.4, 0.2],
        [5.7, 3.8, 1.7, 0.3],
        [5.0, 3.3, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [5.1, 3.8, 1.9, 0.4],
        [4.8, 3.0, 1.4, 0.1],
        [5.0, 3.0, 1.6, 0.2],
        [4.4, 3.2, 1.3, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [4.5, 2.3, 1.3, 0.3],
        [5.1, 3.8, 1.9, 0.4],
        [5.1, 3.5, 1.4, 0.3],
    ];

    let labels = [
        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let learning_rate = 0.1;
    let epochs = 100;

    // Train the model
    let (trained_weights, trained_bias) = train(&inputs, &labels, learning_rate, epochs);

    println!("Trained Weights: {:?}", trained_weights);
    println!("Trained Bias: {:?}", trained_bias);
    /*
    epoch = 10
    Trained Weights: [-0.12457363944662814, -0.31141632749629106, 0.42535906423765324, 0.18659335247085568]
    Trained Bias: -0.06289986228129817

    epoch = 100
    Trained Weights: [-0.3726472362241575, -1.1493118151014219, 1.7181282744314055, 0.7479174088260542]
    Trained Bias: -0.23622522154644904
    */
}

#[test]
fn test_iris_multiclass() {
    const M: usize = 4;
    const N: usize = 30;
    let inputs: [[f64; M]; N] = [
        [5.7, 2.9, 4.2, 1.3],
        [6.9, 3.2, 5.7, 2.3],
        [6.1, 3.0, 4.9, 1.8],
        [6.0, 2.9, 4.5, 1.5],
        [5.4, 3.4, 1.5, 0.4],
        [7.2, 3.0, 5.8, 1.6],
        [6.8, 3.2, 5.9, 2.3],
        [6.9, 3.1, 4.9, 1.5],
        [5.7, 2.9, 4.2, 1.3],
        [5.8, 2.7, 5.1, 1.9],
        [7.7, 2.8, 6.7, 2.0],
        [5.4, 3.0, 4.5, 1.5],
        [5.0, 3.3, 1.4, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.6, 2.9, 3.6, 1.3],
        [5.6, 2.7, 4.2, 1.3],
        [6.4, 2.7, 5.3, 1.9],
        [5.6, 2.8, 4.9, 2.0],
        [6.2, 2.2, 4.5, 1.5],
        [6.1, 2.8, 4.0, 1.3],
        [5.6, 2.9, 3.6, 1.3],
        [6.6, 2.9, 4.6, 1.3],
        [6.5, 3.0, 5.2, 2.0],
        [6.5, 3.2, 5.1, 2.0],
        [5.6, 3.0, 4.5, 1.5],
        [6.3, 3.3, 6.0, 2.5],
        [6.3, 3.3, 6.0, 2.5],
        [5.7, 2.8, 4.1, 1.3],
        [5.0, 3.5, 1.3, 0.3],
        [6.5, 2.8, 4.6, 1.5],
    ];

    //================== LABELS ===========================
    let labels_class0 = [
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ];
    //================== LABELS ===========================
    let labels_class1 = [
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    ];
    //================== LABELS ===========================
    let labels_class2 = [
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let learning_rate = 0.1;
    let epochs = 10;

    // Train the model
    let res = train_multi_class(epochs,
      &inputs, &[labels_class0, labels_class1, labels_class2], learning_rate);

    // println!("Trained Weights & biases: {:?}", res);
    println!("Trained Weights & Biases:");
    for (weights, bias) in &res {
        let weights_str: Vec<String> = weights.iter().map(|w| format!("{:.20}", w)).collect();
        let bias_str = format!("{:.20}", bias);
        println!("Weights: [{}], Bias: {}", weights_str.join(", "), bias_str);
    }

    /*
    Trained Weights & Biases:
    Weights: [-0.14732800490882030919, -0.14235626876994034173, 0.19349365223313150230, 0.13028040185789524497], Bias: -0.05389485934669701467
    Weights: [-0.00005201159237453812, -0.04144377630917304295, -0.01755215146192968048, -0.04331418263230187721], Bias: 0.00710318743326482545
    Weights: [-0.14172799064049107498, 0.02509652620694889646, -0.34834867842714573039, -0.14248617287286668986], Bias: -0.00507673518521851129
 */
}
#[test]
fn test_approx_sigmoid() {
    let x = 1.0;
    let approx_sigmoid_res = approx_sigmoid(x);
    println!("{}", approx_sigmoid_res); // 0.67

    let x = 0.1;
    let approx_sigmoid_res = approx_sigmoid(x);
    println!("{}", approx_sigmoid_res); // 0.517

    let x = -0.5;
    let approx_sigmoid_res = approx_sigmoid(x);
    println!("{}", approx_sigmoid_res); // 0.415

    let x = -1.1;
    let approx_sigmoid_res = approx_sigmoid(x);
    println!("{}", approx_sigmoid_res); // 0.31299999999999994
}
