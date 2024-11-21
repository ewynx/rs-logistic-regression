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

fn compute_gradient(
    features: &[Vec<f64>], // Each row is a data point
    labels: &[f64],        // Corresponding labels
    weights: &[f64],       // Current weights
    bias: f64,             // Current bias
) -> (Vec<f64>, f64) {
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
    (
        weight_gradients.iter().map(|g| g / m).collect(),
        bias_gradient / m,
    )
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

fn train(
    features: &[Vec<f64>],
    labels: &[f64],
    learning_rate: f64,
    epochs: usize,
) -> (Vec<f64>, f64) {
    // Initialize weights and bias within the training function
    let mut weights = vec![0.0; features[0].len()]; // Zero-initialized weights
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

fn main() {
    println!("Hello, world!");
}

#[test]
fn test_setosa_versicolor() {
    // Define the dataset (features and labels)
    let inputs = vec![
      vec![
              5.5,
              2.6,
              4.4,
              1.2,
      ],
      vec![
              5.0,
              3.4,
              1.6,
              0.4,
      ],
      vec![
              5.2,
              4.1,
              1.5,
              0.1,
      ],
      vec![
              6.5,
              2.8,
              4.6,
              1.5,
      ],
      vec![
              5.6,
              3.0,
              4.5,
              1.5,
      ],
      vec![
              4.6,
              3.2,
              1.4,
              0.2,
      ],
      vec![
              6.0,
              2.2,
              4.0,
              1.0,
      ],
      vec![
              5.1,
              3.4,
              1.5,
              0.2,
      ],
      vec![
              6.6,
              2.9,
              4.6,
              1.3,
      ],
      vec![
              4.7,
              3.2,
              1.3,
              0.2,
      ],
      vec![
              6.3,
              3.3,
              4.7,
              1.6,
      ],
      vec![
              5.2,
              3.4,
              1.4,
              0.2,
      ],
      vec![
              5.8,
              2.6,
              4.0,
              1.2,
      ],
      vec![
              6.1,
              2.8,
              4.0,
              1.3,
      ],
      vec![
              5.7,
              2.8,
              4.1,
              1.3,
      ],
      vec![
              5.8,
              4.0,
              1.2,
              0.2,
      ],
      vec![
              4.8,
              3.4,
              1.9,
              0.2,
      ],
      vec![
              5.6,
              3.0,
              4.5,
              1.5,
      ],
      vec![
              4.9,
              3.0,
              1.4,
              0.2,
      ],
      vec![
              5.7,
              3.8,
              1.7,
              0.3,
      ],
      vec![
              5.0,
              3.3,
              1.4,
              0.2,
      ],
      vec![
              6.2,
              2.9,
              4.3,
              1.3,
      ],
      vec![
              5.1,
              3.8,
              1.9,
              0.4,
      ],
      vec![
              4.8,
              3.0,
              1.4,
              0.1,
      ],
      vec![
              5.0,
              3.0,
              1.6,
              0.2,
      ],
      vec![
              4.4,
              3.2,
              1.3,
              0.2,
      ],
      vec![
              4.6,
              3.2,
              1.4,
              0.2,
      ],
      vec![
              4.5,
              2.3,
              1.3,
              0.3,
      ],
      vec![
              5.1,
              3.8,
              1.9,
              0.4,
      ],
      vec![
              5.1,
              3.5,
              1.4,
              0.3,
      ],
];

    let labels = vec![
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
];

    let learning_rate = 0.1;
    let epochs = 1;

    // Train the model
    let (trained_weights, trained_bias) = train(&inputs, &labels, learning_rate, epochs);

    // Collect predictions
    let predictions: Vec<f64> = inputs
        .iter()
        .map(|input| get_prediction(input, &trained_weights, trained_bias))
        .collect();

    // Calculate accuracy
    let accuracy = calculate_accuracy(&predictions, &labels);

    println!("Trained Weights: {:?}", trained_weights);
    println!("Trained Bias: {:?}", trained_bias);
    println!("Predictions: {:?}", predictions);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    /*
        Epochs = 10:
[-0.5252033301021997, -0.3599251494270883, -0.15437820313086892, -0.02547627906962113]
Epochs = 100:
[-0.6926971464326838, -0.4736125559888238, -0.20396579391537806, -0.03248697477463643]
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
