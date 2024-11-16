fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn predict(features: &[f64], weights: &[f64]) -> f64 {
    sigmoid(dot_product(features, weights))
}

fn compute_gradient(
    features: &[Vec<f64>], // Each row is a data point
    labels: &[f64],        // Corresponding labels
    weights: &[f64],       // Current weights
) -> Vec<f64> {
    let m = features.len() as f64;

    // Compute gradient for each weight
    let mut gradients = vec![0.0; weights.len()];
    for i in 0..features.len() {
        let h = sigmoid(dot_product(&features[i], weights));
        let error = h - labels[i];
        for j in 0..weights.len() {
            gradients[j] += error * features[i][j];
        }
    }

    // Average the gradients
    gradients.iter().map(|g| g / m).collect()
}
fn update_weights(weights: &mut [f64], gradients: &[f64], learning_rate: f64) {
    for (w, g) in weights.iter_mut().zip(gradients.iter()) {
        *w -= learning_rate * g;
    }
}
fn train(
    features: &[Vec<f64>],
    labels: &[f64],
    mut weights: Vec<f64>,
    learning_rate: f64,
    epochs: usize,
) -> Vec<f64> {
    for _ in 0..epochs {
        let gradients = compute_gradient(features, labels, &weights);
        update_weights(&mut weights, &gradients, learning_rate);
    }
    weights
}

fn main() {
    println!("Hello, world!");
}

#[test]
fn test_setosa_versicolor() {
    // Define the dataset (features and labels)
let inputs = vec![
        vec![
                5.1,
                3.5,
                1.4,
                0.2,
        ],
        vec![
                4.9,
                3.0,
                1.4,
                0.2,
        ],
        vec![
                4.7,
                3.2,
                1.3,
                0.2,
        ],
        vec![
                4.6,
                3.1,
                1.5,
                0.2,
        ],
        vec![
                5.0,
                3.6,
                1.4,
                0.2,
        ],
        vec![
                5.4,
                3.9,
                1.7,
                0.4,
        ],
        vec![
                4.6,
                3.4,
                1.4,
                0.3,
        ],
        vec![
                5.0,
                3.4,
                1.5,
                0.2,
        ],
        vec![
                4.4,
                2.9,
                1.4,
                0.2,
        ],
        vec![
                4.9,
                3.1,
                1.5,
                0.1,
        ],
        vec![
                5.4,
                3.7,
                1.5,
                0.2,
        ],
        vec![
                4.8,
                3.4,
                1.6,
                0.2,
        ],
        vec![
                4.8,
                3.0,
                1.4,
                0.1,
        ],
        vec![
                4.3,
                3.0,
                1.1,
                0.1,
        ],
        vec![
                5.8,
                4.0,
                1.2,
                0.2,
        ],
        vec![
                5.7,
                4.4,
                1.5,
                0.4,
        ],
        vec![
                5.4,
                3.9,
                1.3,
                0.4,
        ],
        vec![
                5.1,
                3.5,
                1.4,
                0.3,
        ],
        vec![
                5.7,
                3.8,
                1.7,
                0.3,
        ],
        vec![
                5.1,
                3.8,
                1.5,
                0.3,
        ],
        vec![
                5.4,
                3.4,
                1.7,
                0.2,
        ],
        vec![
                5.1,
                3.7,
                1.5,
                0.4,
        ],
        vec![
                4.6,
                3.6,
                1.0,
                0.2,
        ],
        vec![
                5.1,
                3.3,
                1.7,
                0.5,
        ],
        vec![
                4.8,
                3.4,
                1.9,
                0.2,
        ],
        vec![
                5.0,
                3.0,
                1.6,
                0.2,
        ],
        vec![
                5.0,
                3.4,
                1.6,
                0.4,
        ],
        vec![
                5.2,
                3.5,
                1.5,
                0.2,
        ],
        vec![
                5.2,
                3.4,
                1.4,
                0.2,
        ],
        vec![
                4.7,
                3.2,
                1.6,
                0.2,
        ],
];

let labels = vec![
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
];

    // Initialize weights
    let weights = vec![0.0; inputs[0].len()];
    let learning_rate = 0.1;
    let epochs = 10;

    // Train the model
    let trained_weights = train(&inputs, &labels, weights, learning_rate, epochs);

    // Make predictions
    println!("Trained Weights: {:?}", trained_weights);
    for (i, input) in inputs.iter().enumerate() {
        let prediction = predict(input, &trained_weights);
        // println!(
        //     "Data Point: {:?}, True Label: {}, Predicted: {}",
        //     input, labels[i], prediction
        // );
    }
    /*
    Epochs = 10:
Trained Weights: [-0.48510845361963106, -0.3324190006665653, -0.14260313309680092, -0.023492916750428465]
      */
}
